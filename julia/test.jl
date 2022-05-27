using NLsolve, SparseArrays, SparseDiffTools


function objfun!(f,x)
    x = exp.(x)
    f .= 2. .*x .- 10
end

function sparsesolve(f!, x, sp)
    fx = similar(x)
    J = Float64.(sp)
    colors = matrix_colors(J)
    #jac_cache = ForwardColorJacCache(f!,  x, dx=fx; colorvec = colors)
    #jac!(J,x) = forwarddiff_color_jacobian!(J,f!,x,jac_cache)
    jac!(J,x) = forwarddiff_color_jacobian!(J,f!,x,colorvec=colors,sparsity=sp)

    df=OnceDifferentiable(f!,jac!,x,fx,J)
    return nlsolve(df,x)
end

function sparsesolve2(f!, x, sp)
    fx = similar(x)
    J = Float64.(sp)
    colors = matrix_colors(J)
    jac_cache = ForwardColorJacCache(f!,  x, dx=fx; colorvec = colors, sparsity=sp)
    jac!(J,x) = forwarddiff_color_jacobian!(J,f!,x,jac_cache)
    
    df=OnceDifferentiable(f!,jac!,x,fx,J)
    return nlsolve(df,x)
end

# Not added right now
using NonlinearSolve, SciMLNLSolve
function scimlsolve(f!,x,sp;method=NLSolveJL(autodiff=:forward))
    fwrap!(f,x,p) = f!(f,x)
    J = Float64.(sp)
    colors = matrix_colors(J)
    nlf = NonlinearFunction{true}(fwrap!,colorvec=colors);
    nlp = NonlinearProblem{true}(nlf,x);
    return solve(nlp,method)
end



nvec=[2,5,10,100,500,1000,10000]
times=zeros(size(nvec))
allocs=zeros(size(nvec))
for (i,n) in enumerate(nvec)
    x = zeros(n)
    show(x)
    b = @benchmark nlsolve(objfun!,$x,autodiff=:forward)
    times[i] = mean(b.times)/1e6
    allocs[i] = b.allocs
end
return nvec, times, allocs