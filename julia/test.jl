using NLsolve

function objfun!(f,x)
    x = exp.(x)
    f .= 2. .*x .- 10
end

function timestuff()
    nvec=[2,5,10,100,500,1000,10000]
    times=zeros(size(nvec))
    allocs=zeros(size(nvec))
    for (i,n) in enumerate(nvec)
        x = zeros(n)
        show(x)
        b = @benchmark nlsolve(objfun!,x,autodiff=:forward)
        times[i] = mean(b.times)/1e6
        allocs[i] = b.allocs
    end
    return nvec, times, allocs
end