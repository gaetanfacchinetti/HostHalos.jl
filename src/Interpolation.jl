CACHE_LOCATION::String = ".cache/"

GridInterpolator1D{T} = Interpolations.GriddedInterpolation{T, 1, Vector{T}, Interpolations.Gridded{Interpolations.Linear{Interpolations.Throw{Interpolations.OnGrid}}}, Tuple{Vector{T}}}

# tuple of functions to interpolate
const FUNCTIONS_TO_INTERPOLATE = (:ρ_host_spherical, :m_host_spherical, :velocity_dispersion_spherical_kms, :circular_velocity_kms, :σ_baryons, :maximum_impact_parameter, :number_stellar_encounters)

struct HostInterpolation{
    T<:AbstractFloat, 
    P<:HaloProfile,
    BM<:BulgeModel{T}, 
    GMHI<:GasModel{T}, 
    GMH2<:GasModel{T}, 
    SM<:StellarModel{T, <:StellarMassModel{T}}}

    interp_map::NamedTuple{FUNCTIONS_TO_INTERPOLATE, NTuple{7, GridInterpolator1D{T}}}
    log10_r_min::T
    log10_r_max::T
    host::HostModel{T, P, BM, GMHI, GMH2, SM}

end


const HostInterpolationType{T} = HostInterpolation{
    T,
    <:HaloProfile,
    <:BulgeModel{T},
    <:GasModel{T},
    <:GasModel{T},
    <:StellarModel{T, <:StellarMassModel{T}}
}


function get_hash(host::HostModel{T, P, BM, GMHI, GMH2, SM}) where {T<:AbstractFloat, P<:HaloProfile, BM<:BulgeModel{T}, GMHI<:GasModel{T},  GMH2<:GasModel{T},  SM<:StellarModel{T, <:StellarMassModel{T}}} 
    
    if T === Float64
        return string(hash((host.name * "_f64")), base=16)
    elseif T === Float32
        return string(hash((host.name * "_f32")), base=16)
    end

    throw("No good hash value for type T")

end

function get_filename(host::HostModelType{T}, s::Symbol, str::String = "") where {T<:AbstractFloat}
    
    (str != "") && (str = "_" * str )

    !(isdir(CACHE_LOCATION)) && mkdir(CACHE_LOCATION)
    filenames  = readdir(CACHE_LOCATION)
    file       = string(s) * str *  "_" * get_hash(host) * ".jld2" 

    return CACHE_LOCATION * file, (file in filenames)

end



# save the different functions in the folder
function save_host_functions(host::HostModelType{T}, s::Symbol) where {T<:AbstractFloat}

    # Check if the file already exists
    filename, exist = get_filename(host, s)
    (exist) && return true
    
    r = exp10.(range(log10(T(1e-3) * host.halo.rs), log10(host.rt), 500))

    @info "| Saving " * string(s) * " in cache" 

    y = Vector{T}(undef, 500)
    for ir in 1:500
        y[ir] = @eval $s($(Ref(r))[][$(Ref(ir))[]], $(Ref(host))[])
    end

    JLD2.jldsave(filename; r = r, y = y)

    return true

end


function HostInterpolation(host::HostModelType{T}) where {T<:AbstractFloat}
    
    log10_r_min = log10_r_max = T(0)
    interp_pairs = Vector{Pair{Symbol, GridInterpolator1D{T}}}()

    for s ∈ FUNCTIONS_TO_INTERPOLATE

        filename, exist = get_filename(host, s)
        !exist && save_host_functions(host, s)

        log10_r, log10_y = let
            JLD2.jldopen(filename, "r") do file
                log10.(file["r"]), 
                log10.(file["y"])
            end
        end

       push!(interp_pairs, s => Interpolations.interpolate((log10_r,), log10_y,  Interpolations.Gridded(Interpolations.Linear())))

       log10_r_min = minimum(log10_r)
       log10_r_max = maximum(log10_r)
    
    end

    interp_map = (; interp_pairs...)

    return HostInterpolation(interp_map, log10_r_min, log10_r_max, host)

end


for f ∈ FUNCTIONS_TO_INTERPOLATE[1:end-1]
    @eval begin
        function ($f)(r::T, wrapper::HostInterpolationType{T}) where {T<:AbstractFloat}

            log10_r = log10(r)
            
            if (log10_r > wrapper.log10_r_min) && (log10_r < wrapper.log10_r_max)
                return exp10(wrapper.interp_map.$f(log10_r))
            end
            
            return $f(r, wrapper.host)
        end
    end
end


function number_stellar_encounters(r::T, wrapper::HostInterpolationType{T}) where {T<:AbstractFloat}
    
    log10_r = log10(r)
    
    if (log10_r > wrapper.log10_r_min) && (log10_r < wrapper.log10_r_max)
        return floor(Int, exp10(wrapper.interp_map.number_stellar_encounters(log10_r)))
    end

    return number_stellar_encounters(r, wrapper.host)
end