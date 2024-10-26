using PyPlot: matplotlib
const mpl = matplotlib
const plt = matplotlib.pyplot
const axes_grid1 = pyimport("mpl_toolkits.axes_grid1")
plt.style.use("bmh")
function subplots(args...; kwargs...)
    retval = plt.subplots(args...; tight_layout=true, kwargs...)
    plt.close(first(retval))
    return retval
end
function savefig(fig, fn; kwargs...)
    fig.savefig(fn; bbox_inches="tight", kwargs...)
end

const pd = pyimport("pandas")

"check if the current program is running in IJulia"
isijulia() = (isdefined(Main, :IJulia) && Main.IJulia.inited)

"check if the current program is running with Distributed"
isdistributed() = isdefined(Main, :Distributed)

"pmap if Distribued is used"
maybepmap(args...; kwargs...) = 
    (isdistributed() ? pmap(args...; kwargs...) : map(args...; kwargs...))


# create a list of help dir functions
# ref: https://github.com/JuliaDynamics/DrWatson.jl/blob/5667d46ca1e7d9feb567484e9d777bc0b1519a73/src/project_setup.jl#L37 
for dir_type in ("artifacts", "results", "log")
    function_name = Symbol(dir_type * "dir")
    @eval begin
        $function_name(args...; suffix="") = projectdir($dir_type, args...) * (isempty(suffix) ? "" : "-$suffix")
    end
end

# https://github.com/comonicon/Comonicon.jl/blob/f1e069cb4bb7e457c867fd6067ce05d6c7ff5963/src/builder/install.jl#L171
"create path if not exist"
function ensure_path(path)
    if !ispath(path)
        @info "creating $path"
        mkpath(path)
    end
    return path
end

function make_Z(f, is=nothing, js=nothing; xlim=(-6, 2), ylim=(-2, 2))
    is = isnothing(is) ? (xlim[1]:0.1:xlim[2]) : is
    js = isnothing(js) ? (ylim[1]:0.1:ylim[2]) : js
    X = Matrix{Float64}(undef, length(is), length(js))
    Y = Matrix{Float64}(undef, length(is), length(js))
    Z = Matrix{Float64}(undef, length(is), length(js))
    for (i, iv) in enumerate(is), (j, jv) in enumerate(js)
        X[i,j] = iv
        Y[i,j] = jv
        Z[i,j] = f([iv, jv])
    end
    return X, Y, Z
end

function plot_contour!(ax, X, Y, Z, label=nothing; makebar=true, annotate_at=nothing, force_lims=false, make_equal=true, kwargs...)
    contour = ax.contour(X, Y, Z, alpha=0.7; kwargs...)
    if makebar
        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(contour, cax=cax, ax=ax, orientation="vertical")
        if !isnothing(annotate_at)
            xloc = -0.01
            yloc = annotate_at
            cax.annotate("", xy=(xloc, yloc), xytext=(xloc - 0.03, yloc), arrowprops=Dict("headwidth" => 5.0, "color" => "red", "alpha" => 0.7))
        end
        isnothing(label) || cbar.set_label(label)
    end
    if force_lims
        ax.set_xlim(extrema(X)...)
        ax.set_ylim(extrema(Y)...)
    else
        make_equal && ax.axis("equal")
    end
end

using Match

function target_dim_by_name(target_name)
    return @match target_name begin
        :funnel || :gaussian => 2
        :funnel3 => 3
        :funnel5 => 5
        :funnel11 => 11
        :funnel21 => 21
        :funnel51 => 51
        :funnel101 => 101
        _ => @error("undefined dim for $target_name")
    end
end