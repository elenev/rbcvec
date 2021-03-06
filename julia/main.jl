using Interpolations, NLsolve, SparseArrays, SparseDiffTools
import QuantEcon.rouwenhorst

Base.@kwdef struct Params
    β::Float64 = 0.99
    α::Float64 = 0.33
    δ::Float64 = 0.1
    γ::Float64 = 2.
    ϕ::Float64 = 3.
end

function sparse_nlsolve(f!, x; sparsity=sparse(1:length(x),1:length(x),true), colorvec=matrix_colors(sparsity))
    dx = similar(x)
    J = Float64.(sparsity)
    jac_cache = ForwardColorJacCache(f!, Array(x), 2; dx, colorvec, sparsity)
    jac!(J,x) = forwarddiff_color_jacobian!(J,f!,x,jac_cache)
    df=OnceDifferentiable(f!,jac!,x,dx,J)
    return nlsolve(df,x)
end

function main(;Na = 11, Nk = 200, maxit = 500, mode=:vec)
    
    p = Params()

    function uprime!(MU,c)
        @. MU = c^(-p.γ)
    end

    function uprime(c)
        margUtil = similar(c)
        uprime!(margUtil,c)
        return margUtil
    end
    
    function uprime(c::Number)
        return c^(-p.γ)
    end

    # Exog environment
    function exog_env()
        μ = 0
        σ = 0.02
        ρ = 0.8

        mc = rouwenhorst(Na,ρ,σ*sqrt(1-ρ^2),μ)
        return mc.p, mc.state_values
    end
    
    P, a_grid = exog_env()
    #piprob = P^500
    #piprob = piprob[1:1,:]
    #check = sqrt( (piprob * a_grid.^2)[] .- (piprob * a_grid)[]^2)

    # Steady State
    kss = ( (1 - p.β*(1-p.δ))/(p.α .* p.β) ) ^ ( 1/(p.α-1) )
    css = kss^p.α - p.δ * kss
    qss = p.β*p.α*kss^(p.α-1) / (1 - p.β*(1-p.δ)) # should be 1

    # Endog grid
    k_grid = exp.( LinRange(-0.2, 0.2, Nk) )' * kss

    # Reshape stuff for vectorized functions
    P3 = reshape(P,(Na,1,Na))
    a_next = reshape(a_grid,(1,1,Na))
    A = repeat(a_grid,1,Nk,Na)

    # Interpolants
    state_space = (a_grid, k_grid[:])
    Kupd = repeat(k_grid,Na,1)
    logC = log(css)*ones(Na,Nk)
    logQ = log(qss)*ones(Na,Nk)
    fK = LinearInterpolation(state_space, Kupd, extrapolation_bc = Line())
    fQ = LinearInterpolation(state_space, logQ, extrapolation_bc = Line())
    fC = LinearInterpolation(state_space, logC, extrapolation_bc = Line())

    # Precomputed output (constant at a given grid point)
    Y = exp.(a_grid) .* k_grid.^p.α

    # Solver options

    # Preallocate some matrices
    Kp = copy(Kupd)
    MU = zeros(Na,Nk,Na)
    RHS = zeros(Na,Nk)
    Qnext = similar(MU)
    Cnext = similar(MU)
    tmp = zeros(Na,Nk,Na)

    # P transpose
    # Pt = P'

    # Sparsity pattern
    sp = kron( ones(2,2), sparse(1:Nk*Na, 1:Nk*Na,true) )
    colors = matrix_colors(sp)

    # Thread pattern
    thread_pattern = [1:2, 3:4, 5:6, 7:8, 9:10, [11]]
    #thread_pattern = [[1],[2],[3],4:5]
    thread_sp = map( idc -> kron( ones(2,2), sparse(1:Nk*length(idc), 1:Nk*length(idc),true) ), thread_pattern)
    thread_colors = map( matrix_colors, thread_sp) 

    # Iterate
    iter = 1

    function update_point(a, k, pnext, Knext_row, Cnext_row, Qnext_row, guess)
        mu_next = uprime(Cnext_row)
        rhs_tilde = @. p.β * mu_next * (p.α*exp(a_grid) * Knext_row^(p.α-1) + (1-p.δ)*Qnext_row)
        rhs = (pnext * rhs_tilde)[]

        function solveEqm!(fx,x)
            c = exp(x[1])
            q = exp(x[2])
            fx[1] = c^(-p.γ) * q - rhs
            fx[2] = 1 + p.ϕ*(Knext_row/k-1) - q
        end

        #test = zeros(2);
        #solveEqm!(test,guess)
        #display(test)

        result = nlsolve(solveEqm!, guess)
        if !converged(result)
            println("No solution in iteration $iter at point ($a,$k)")
        end
        logc = result.zero[1]
        logq = result.zero[2]

        kupd = exp(a)*k^p.α + (1-p.δ)*k - exp(logc)
        return kupd, logc, logq
    end

    function update!(Kupd,logC,logQ,Cnext,Qnext)
        uprime!(MU,Cnext)
        @. tmp = P3 * p.β * MU * (p.α*exp(a_next) * Kp^(p.α-1) + (1-p.δ)*Qnext)
        #@tturbo inline=true tmp2 = P3 .* p.β .* MU .* (p.α.*exp.(a_next) .* Kp.^(p.α-1) + (1-p.δ).*Qnext)
        #println("tmp[end,end,end] is $(tmp[end,end,end]) but tmp2[end,end,end] is $(tmp2[end,end,end]).")
        #println("Full comparison returns: $(tmp ≈ tmp2)")
        RHS .= dropdims( sum( tmp, dims=3), dims=3)

        #@inbounds for j=1:Nk, i=1:Na
        #   RHS[i,j] = P[i,:]' * (p.β .* MU[i,j,:] .* (p.α.*exp.(a_grid) .* Kp[i,j].^(p.α-1) + (1-p.δ).*Qnext[i,j,:]) )
        #end

        #@inbounds Threads.@threads for i=1:Na
        #   RHS[i,:] = (p.β .* MU[i,:,:] .* (p.α.*exp.(a_grid') .* Kp[i,:].^(p.α-1) + (1-p.δ).*Qnext[i,:,:]) ) * Pt[:,i]
        #end

        function equilibrium!(fx1,fx2,c,q)
            uprime!(fx1,c)
            @. fx1 = fx1 * q - RHS
            @. fx2 = 1 + p.ϕ*(Kp/k_grid-1) - q
        end

        function alloc!(c,q,x)
            c .= reshape( x[1:Na*Nk], Na, Nk)
            q .= reshape( x[Na*Nk+1:end], Na, Nk)
        end

        function alloc(x)
            x1 = reshape(view(x,1:Na*Nk),Na,Nk)
            x2 = reshape(view(x,Na*Nk+1:2*Na*Nk),Na,Nk)
            return x1, x2
        end

        function solveEqm!(fx,x)
            logc, logq = alloc(x)
            c = exp.(logc)
            q = exp.(logq)
            err1, err2 = alloc(fx)
            equilibrium!(err1,err2,c,q)
        end
        
        guess = vcat(logC[:], logQ[:])
        #err = similar(guess)
        #solveEqm!(err,guess)

        #result = nlsolve(solveEqm!, guess, autodiff = :forward)
        result = sparse_nlsolve(solveEqm!, guess; sparsity=sp, colorvec=colors)
        #show(result.f_calls)
        alloc!(logC, logQ, result.zero)

        @. Kupd = Y + (1-p.δ)*k_grid - exp(logC)

        return converged(result)
        #return true
    end

    function update_thread!(Kupdt,logCt,logQt,Cnextt,Qnextt,t,spt,colorst)
        Nat = size(logCt,1)
        MUt = uprime(Cnextt)
        Pt = P3[t,:,:]
        Kpt = Kp[t,:]
        tmpt = @. Pt * p.β * MUt * (p.α*exp(a_next) * Kpt^(p.α-1) + (1-p.δ)*Qnextt)

        RHSt = dropdims( sum( tmpt, dims=3), dims=3)

        function equilibrium!(fx1,fx2,c,q)
            uprime!(fx1,c)
            @. fx1 = fx1 * q - RHSt
            @. fx2 = 1 + p.ϕ*(Kpt/k_grid-1) - q
        end

        function alloc!(c,q,x)
            c .= reshape( x[1:Nat*Nk], Nat, Nk)
            q .= reshape( x[Nat*Nk+1:end], Nat, Nk)
        end

        function alloc(x)
            x1 = reshape(view(x,1:Nat*Nk),Nat,Nk)
            x2 = reshape(view(x,Nat*Nk+1:2*Nat*Nk),Nat,Nk)
            return x1, x2
        end

        function solveEqm!(fx,x)
            logc, logq = alloc(x)
            c = exp.(logc)
            q = exp.(logq)
            err1, err2 = alloc(fx)
            equilibrium!(err1,err2,c,q)
        end
        
        guess = vcat(logCt[:], logQt[:])


        result = sparse_nlsolve(solveEqm!, guess; sparsity=spt, colorvec=colorst)
        alloc!(logCt, logQt, result.zero)

        @. Kupdt = Y[t,:] + (1-p.δ)*k_grid - exp(logCt)


        return converged(result)
    end


    function iterate!(fK,fQ,fC,Kupd;mode=:vec)
        # Get next period Q and C
        Kp .= Kupd
        KKp = repeat(Kp,1,1,Na)
        Qnext = exp.( fQ.(A,KKp) )
        Cnext = exp.( fC.(A,KKp) )

        # Iterate
        if mode == :vec 
            conv = update!(Kupd,logC,logQ,Cnext,Qnext)
            if !conv
                println("No solution in iteration $iter")
            end
        elseif mode == :element
            for (j,k) in enumerate(k_grid), (i, a) in enumerate(a_grid)
                guess = [logC[i,j], logQ[i,j]]
                Kupd[i,j], logC[i,j], logQ[i,j] = 
                    update_point(a,k,P[i:i,:],Kp[i,j],Cnext[i,j,:],Qnext[i,j,:], guess)
            end
        elseif mode == :threadvec
            conv = similar(thread_pattern, Bool)
            Threads.@threads for i in eachindex(thread_pattern)
                t = thread_pattern[i]
                conv = update_thread!(view(Kupd,t,:), view(logC,t,:), view(logQ,t,:), view(Cnext,t,:,:), view(Qnext,t,:,:), t,thread_sp[i],thread_colors[i])
                if !conv
                    println("No solution in iteration $iter, exog states $t")
                end
                #display(Kupd[t,:])
            end
        end

        # Update Interpolants
        fK.itp.coefs .= Kupd
        #fK .= LinearInterpolation(state_space, Kupd, extrapolation_bc = Line())
        fQ.itp.coefs .= logQ
        fC.itp.coefs .= logC

        #display(Kupd)

        # Check and report convergence
        dist = log10( maximum( abs.(Kupd-Kp)) )
        println("--- Iteration $iter: Dist = $dist")

        return dist
    end

    #iterate!()
    t0 = time()
    while iter <= maxit
        dist = iterate!(fK,fQ,fC,Kupd;mode)
        if dist < -5
            break
        else
            iter += 1
        end
    end
    t1 = time()
    out = (t1-t0) / iter

    return (; fK, fC, fQ), out

end

sol, t = main(mode = :vec); print(1000 * t)
