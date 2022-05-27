using Interpolations, NLsolve
import QuantEcon.rouwenhorst

Base.@kwdef struct Params
    β::Float64 = 0.99
    α::Float64 = 0.33
    δ::Float64 = 0.1
    γ::Float64 = 2.
    ϕ::Float64 = 3.
end

function main(;Na = 11, Nk = 200, maxit = 500)
    
    p = Params()

    function uprime!(MU,c)
        MU .= c.^(-p.γ)
    end

    function uprime(c)
        margUtil = similar(c)
        uprime!(margUtil,c)
        return margUtil
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
    piprob = P^500
    piprob = piprob[1:1,:]
    check = sqrt( (piprob * a_grid.^2)[] .- (piprob * a_grid)[]^2)

    # Steady State
    kss = ( (1 - p.β*(1-p.δ))/(p.α .* p.β) ) ^ ( 1/(p.α-1) )
    css = kss^p.α - p.δ * kss
    qss = p.β*p.α*kss^(p.α-1) / (1 - p.β*(1-p.δ))

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

    # Iterate
    iter = 1

    function update!(Kupd,logC,logQ,Cnext,Qnext)
        uprime!(MU,Cnext)
        RHS .= dropdims( sum( 
            P3 .* p.β .* MU .* (p.α*exp.(a_next) .* Kp.^(p.α-1) + (1-p.δ).*Qnext),
             dims=3), dims=3)

        function equilibrium!(fx1,fx2,c,q)
            uprime!(fx1,c)
            fx1 .= fx1 .* q .- RHS
            fx2 .= 1 .+ p.ϕ*(Kp./k_grid.-1) .- q
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
        err = similar(guess)
        solveEqm!(err,guess)

        result = nlsolve(solveEqm!, guess, autodiff = :forward)
        show(result.f_calls)
        alloc!(logC, logQ, result.zero)

        Kupd .= Y .+ (1-p.δ).*k_grid - exp.(logC)

        return converged(result)
        #return true
    end


    function iterate!()
        # Get next period Q and c
        Kp .= Kupd
        KKp = repeat(Kp,1,1,Na)
        Qnext = exp.( fQ.(A,KKp) )
        Cnext = exp.( fC.(A,KKp) )

        # Iterate
        conv = update!(Kupd,logC,logQ,Cnext,Qnext)
        if !conv
            println("No solution in iteration $iter")
        end

        # Update Interpolants
        fK.itp.coefs .= Kupd
        #fK .= LinearInterpolation(state_space, Kupd, extrapolation_bc = Line())
        fQ.itp.coefs .= logQ
        fC.itp.coefs .= logC

        # Check and report convergence
        dist = log10( maximum( abs.(Kupd-Kp)) )
        println("--- Iteration $iter: Dist = $dist")

        return dist
    end

    #iterate!()
    t0 = time()
    while iter <= maxit
        dist = iterate!()
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

sol, t = main()
print(1000 * t)