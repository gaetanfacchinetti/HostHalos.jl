module HostHalos

import CosmoTools: BkgCosmology, planck18_bkg, lookback_redshift, δt_s, z_to_a, Cosmology, planck18, MPC_TO_KM, MYR_TO_S, G_NEWTON
import CosmoTools: Halo, HaloProfile, nfwProfile, αβγProfile, m_halo, ρ_halo, μ_halo, coreProfile, rΔ

import QuadGK, JLD2,  Interpolations, SpecialFunctions

export ρ_HI, ρ_H2, ρ_ISM, ρ_baryons, ρ_baryons_spherical, host_halo, m_baryons_spherical
export m_host_spherical, μ_host_spherical, ρ_host_spherical, σ_baryons, age_host
export HostModel, BulgeModel
export circular_velocity, circular_period, number_circular_orbits, velocity_dispersion_spherical
export circular_velocity_kms, velocity_dispersion_spherical_kms 
export milky_way_MM17_g1, milky_way_MM17_g0
export maximum_impact_parameter, number_stellar_encounters, stellar_mass_function, stellar_mass_model_C03, moments_stellar_mass
export load!

abstract type BulgeModel{T<:Real} end
abstract type GasModel{T<:Real} end
abstract type StellarModel{T<:Real} end

mutable struct HostModel{T<:Real}

    const name::String
    const halo::Halo{T}
    const bulge::BulgeModel{T}
    const gas_HI::GasModel{T}
    const gas_H2::GasModel{T}
    const stars::StellarModel{T}

    const age::T # in s
    const rt::T # in Mpc

    ## precomputed tabulated functions
    ρ_host_spherical::Union{Nothing, Function}
    m_host_spherical::Union{Nothing, Function}
    velocity_dispersion_spherical_kms::Union{Nothing, Function}
    circular_velocity_kms::Union{Nothing, Function}
    σ_baryons::Union{Nothing, Function}
    maximum_impact_parameter::Union{Nothing, Function}
    number_stellar_encounters::Union{Nothing, Function}

end

Base.length(::HostModel) = 1
Base.iterate(iter::HostModel) = (iter, nothing)
Base.iterate(::HostModel, state::Nothing) = nothing

function HostModel(name::String, halo::Halo{<:Real}, bulge::BulgeModel{<:Real}, 
    gas_HI::GasModel{<:Real}, gas_H2::GasModel{<:Real}, 
    stars::StellarModel{<:Real}, age::Real,
    rt::Real = -1)

    # by default the tidal radius is set to the virial radius
    (rt == -1) && (rt = rΔ(halo))

    return HostModel(name, halo, bulge, gas_HI, gas_H2, stars, age, rt, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

function Base.getproperty(obj::HostModel, s::Symbol)

    # we load the data if necessary
    if getfield(obj, s) === nothing
        setfield!(obj, s, _load(obj, s))
    end

    return getfield(obj, s)
end


# overrinding print function
Base.show(io::IO, host::HostModel) = print(io, "host name : " * host.name)


ρ_HI(r::Real, z::Real, host::HostModel) = ρ_gas(r, z, host.gas_HI)
ρ_H2(r::Real, z::Real, host::HostModel) = ρ_gas(r, z, host.gas_H2)
ρ_ISM(r::Real, z, host::HostModel) = ρ_HI(r, z, host) + ρ_H2(r, z, host) 
ρ_bulge(r::Real, z::Real, host::HostModel) = ρ_bulge(r, z, host.bulge)
ρ_stars(r::Real, z::Real, host::HostModel) = ρ_stars(r, z, host.stars)
ρ_baryons(r::Real, z::Real, host::HostModel) = ρ_ISM(r, z, host) + ρ_bulge(r, z, host) + ρ_stars(r, z, host) 

# Surface density (for disc galaxies)
σ_HI(r::Real, host::HostModel) = σ_gas(r, host.gas_HI)
σ_H2(r::Real, host::HostModel) = σ_gas(r, host.gas_H2)
σ_ISM(r::Real, host::HostModel) = σ_HI(r, host) + σ_H2(r, host)
σ_bulge(r::Real, host::HostModel) = σ_bulge(r, host.bulge)
σ_stars(r::Real, host::HostModel) = σ_stars(r, host.stars)
σ_baryons(r::Real, host::HostModel) = σ_ISM(r, host) + σ_bulge(r, host) + σ_stars(r, host) 


""" age of the host in s at any redshift """
function age_host(z::Real, host::HostModel, cosmo::BkgCosmology = planck18_bkg; kws...)
    z_max = lookback_redshift(host.age)
    return δt_s(z_to_a(z_max), z_to_a(z), cosmo; kws...)
end

###########################
## SPHERICISED QUANTITIES

function der_ρ_baryons_spherical(xp::Real, r::Real, host::HostModel)

    (ρ_baryons == 0) && (return 0.0)

    y = sqrt(1 - xp^2)
    return xp^2/y * (ρ_baryons(r * xp, - r * y, host) + ρ_baryons(r * xp, r * y, host))/2.0
end

ρ_baryons_spherical(r::Real, host::HostModel) = QuadGK.quadgk(xp -> der_ρ_baryons_spherical(xp, r, host), 0, 1, rtol=1e-4)[1] 
m_baryons_spherical(r::Real, host::HostModel) = 4.0 * π * QuadGK.quadgk(rp -> rp^2 * ρ_baryons_spherical(rp, host), 0, r, rtol=1e-3)[1] 

ρ_host_spherical(r::Real, host::HostModel) = ρ_baryons_spherical(r, host) + ρ_halo(r, host.halo)
m_host_spherical(r::Real, host::HostModel) = m_baryons_spherical(r, host) + m_halo(r, host.halo)

""" total mass of the host """
m_host(host::HostModel) = host.m_host_spherical === nothing ? m_host_spherical(host.rt, host) : host.m_host_spherical(host.rt)

mΔ(h::Halo{<:Real}, Δ::Real = 200, cosmo::Cosmology = planck18) = mΔ(h, Δ, cosmo.bkg.ρ_c0)

""" circular velocity in (Mpc / s) for `r` in Mpc """
circular_velocity(r::Real, host::HostModel = milky_way_MM17_g1) = sqrt(G_NEWTON * m_host_spherical(r, host) / r) 

""" circular velocity in (km / s) for `r` in Mpc """
circular_velocity_kms(r::Real, host::HostModel = milky_way_MM17_g1) = circular_velocity(r, host) *  MPC_TO_KM

""" circular period in s for `r` in Mpc """
circular_period(r::Real, host::HostModel = milky_way_MM17_g1) = 2.0 * π * r / circular_velocity(r, host)  

""" number or circular orbits with `r` in Mpc """
number_circular_orbits(r::Real, host::HostModel = milky_way_MM17_g1, z::Real = 0, bkg_cosmo::BkgCosmology = planck18_bkg; kws...) = floor(Int, age_host(z, host, bkg_cosmo, kws...) / circular_period(r, host))

""" Jeans dispersion in (Mpc / s)"""
velocity_dispersion_spherical(r::Real, host::HostModel = milky_way_MM17_g1) = (r >= host.rt) ? 0.0 : sqrt(G_NEWTON / ρ_halo(r, host.halo) * QuadGK.quadgk(rp -> ρ_halo(rp, host.halo) * m_host_spherical(rp, host)/rp^2, r, host.rt, rtol=1e-3)[1])

""" Jeans dispersion in (km / s)"""
velocity_dispersion_spherical_kms(r::Real, host::HostModel = milky_way_MM17_g1) = velocity_dispersion_spherical(r, host) * MPC_TO_KM




##################################################
# Stellar mass and relative velocity distributions
function stellar_mass_function_LNPL(m::Real, m_cut::NTuple{4, <:Real}, norm::NTuple{4, <:Real}, indices::NTuple{3, <:Real}, med, σm)
    
    (log10(m) <= m_cut[1]) && (return norm[1] * exp(-(log10(m) - log10(med))^2 / (2. * (σm)^2)) / m) 
    (m_cut[1] < log10(m) && log10(m) <= m_cut[2]) && (return norm[2] *  m^indices[1])
    (m_cut[2] < log10(m) && log10(m) <= m_cut[3]) && (return norm[3] *  m^indices[2])
    (m_cut[3] < log10(m) && log10(m) <= m_cut[4]) && (return norm[4] *  m^indices[3])

    return 0
end


moments_stellar_mass_LNPL(n::Int, m_cut::NTuple{4, <:Real}, norm::NTuple{4, <:Real}, indices::NTuple{3, <:Real}, med, σm) = QuadGK.quadgk(lnm -> exp(lnm)^(n+1) * stellar_mass_function_LNPL(exp(lnm), m_cut, norm, indices, med, σm), log(1e-7), log(10.0^1.8), rtol=1e-10)[1]  

abstract type StellarMassModel{T<:Real} end

""" Log-Lormal and Power-Law Mass model"""
struct LNPLStellarMassModel{T<:Real} <: StellarMassModel{T}
    
    m_cut::NTuple{4, T} # in Msun
    norm::NTuple{4, T}
    indices::NTuple{3, T}
    med::T
    σm::T

    average_mstar::Real   # in Msun
    average_mstar2::Real  # in Msun^2

end

function LNPLStellarMassModel(m_cut::NTuple{4, <:Real}, norm::NTuple{4, <:Real}, indices::NTuple{3, <:Real}, med::Real, σm::Real, av_mstar::Union{Real, Nothing} = nothing, av_mstar2::Union{Real, Nothing} = nothing)
    
    # here we precompte the first and second moment of the stellar mass function if they are not given in input
    _av_mstar = (av_mstar === nothing) ? moments_stellar_mass_LNPL(1, m_cut, norm, indices, med, σm) : av_mstar
    _av_mstar2 = (av_mstar === nothing) ? moments_stellar_mass_LNPL(2, m_cut, norm, indices, med, σm) : av_mstar2
    
    return LNPLStellarMassModel(m_cut, norm, indices, med, σm, _av_mstar, _av_mstar2)
end

const stellar_mass_model_C03::LNPLStellarMassModel = LNPLStellarMassModel((0.0, 0.54, 1.26, 1.80), (0.2613, 0.0728, 0.02481, 0.0004135), (-5.37, -4.53, -3.11), 0.079, 0.69, 0.16794677064645963, 0.08941378784419536)


stellar_mass_function(m::Real, model::LNPLStellarMassModel{<:Real} = stellar_mass_model_C03) = stellar_mass_function_LNPL(m, model.m_cut, model.norm, model.indices, model.med, model.σm)
moments_stellar_mass(n::Int, model::LNPLStellarMassModel{<:Real}= stellar_mass_model_C03) =  moments_stellar_mass_LNPL(n, model.m_cut, model.norm, model.indices, model.med, model.σm)


""" maximum impact parameters at distance r (Mpc) from the Galactic center"""
function maximum_impact_parameter(r::Real, host::HostModel{<:Real})
    return host.stars.mass_model.average_mstar^(1/3)/σ_stars(r, host) * QuadGK.quadgk(lnz -> exp(lnz) * ρ_stars(r, exp(lnz), host)^(2/3), log(1e-10), log(1e+0), rtol=1e-10)[1] 
end

""" number of stars encountered on one disk crossing """
function number_stellar_encounters(r::Real, host::HostModel{<:Real}, θ::Real = π/3.0)
    return floor(Int, σ_stars(r, host) / host.stars.mass_model.average_mstar * π / cos(θ) * maximum_impact_parameter(r, host)^2)
end


###########################
## McMillan 2017 Model


# Definition of the profiles
function ρ_spherical_BG02(r::Real, z::Real; ρ0b::Real, r0::Real, q::Real, α::Real, rcut::Real) 
    rp = sqrt(r^2 + (z/q)^2)
    return ρ0b/((1+rp/r0)^α) * exp(-(rp/rcut)^2)
end

ρ_exponential_disc(r::Real, z::Real; σ0::Real, rd::Real, zd::Real) = σ0/(2.0*zd)*exp(-abs(z)/zd - r/rd)
σ_exponential_disc(r::Real; σ0::Real, rd::Real) = σ0 * exp(-r/rd)
ρ_sech_disc(r::Real, z::Real; σ0::Real, rd::Real, rm::Real, zd::Real) = σ0/(4.0*zd)*exp(-rm/r - r/rd)*(sech(z/(2.0*zd)))^2
σ_sech_disc(r::Real; σ0::Real, rd::Real, rm::Real, zd::Real)= σ0 * exp(-r/rd - rm/r)

struct AxiSymmetricBGBulgeModel{T<:Real} <: BulgeModel{T}
    ρ0b::T
    r0::T
    q::T
    α::T
    rcut::T
end

ρ_bulge(r::Real, z::Real, bulge::AxiSymmetricBGBulgeModel) = ρ_spherical_BG02(r, z, ρ0b = bulge.ρ0b, r0 = bulge.r0, q = bulge.q, α = bulge.α, rcut = bulge.rcut)
σ_bulge(r::Real, bulge::AxiSymmetricBGBulgeModel) = QuadGK.quadgk( z-> ρ_bulge(r, z, bulge), -10*bulge.q/r, 10*bulge.q/r, rtol=1e-3)[1]

struct SechGasModel{T<:Real} <: GasModel{T}
    σ0::T
    zd::T
    rm::T
    rd::T
end

ρ_gas(r::Real, z::Real, gas::SechGasModel) = ρ_sech_disc(r, z, σ0 = gas.σ0, rd = gas.rd, rm = gas.rm, zd = gas.zd)
σ_gas(r::Real, gas::SechGasModel) = σ_sech_disc(r, σ0 = gas.σ0, rd = gas.rd, rm = gas.rm, zd = gas.zd)

struct DoubleDiscStellarModel{T<:Real} <: StellarModel{T}
    mass_model::StellarMassModel{T}
    thin_σ0::T
    thin_zd::T
    thin_rd::T
    thick_σ0::T
    thick_zd::T
    thick_rd::T  
end

DoubleDiscStellarModel(mass_model::StellarMassModel{<:Real}, thin_σ0::Real, thin_rd::Real, thin_zd::Real, thick_σ0::Real, thick_rd::Real, thick_zd::Real) = DoubleDiscStellarModel(mass_model, promote(thin_σ0, thin_rd, thin_zd, thick_σ0, thick_rd, thick_zd)...)
ρ_stars(r::Real, z::Real, stars::DoubleDiscStellarModel) = ρ_exponential_disc(r, z, σ0 = stars.thick_σ0, zd = stars.thick_zd, rd = stars.thick_rd) + ρ_exponential_disc(r, z, σ0 = stars.thin_σ0, zd = stars.thin_zd, rd = stars.thin_rd)
σ_stars(r::Real, stars::DoubleDiscStellarModel) = σ_exponential_disc(r, σ0 = stars.thick_σ0, rd = stars.thick_rd) + σ_exponential_disc(r, σ0 = stars.thin_σ0, rd = stars.thin_rd)

const bulge_MM17::AxiSymmetricBGBulgeModel = AxiSymmetricBGBulgeModel(9.73e+19, 7.5e-5, 0.5, 1.8, 2.1e-3)
const gas_HI_MM17::SechGasModel = SechGasModel(5.31e+13, 8.5e-5, 4.0e-3, 7.0e-3)
const gas_H2_MM17::SechGasModel = SechGasModel(2.180e+15, 4.5e-5, 1.2e-2, 1.5e-3) 

milky_way_MM17_g1 = HostModel("MilkyWay_MM17_g1", Halo(nfwProfile, 9.24412866426226e+15, 1.86e-2),  bulge_MM17, gas_HI_MM17, gas_H2_MM17, DoubleDiscStellarModel(stellar_mass_model_C03, 8.87e+14, 3.0e-4, 2.53e-3, 1.487e+14, 9.0e-4, 3.29e-3), 1e+4 * MYR_TO_S)
milky_way_MM17_g0 = HostModel("MilkyWay_MM17_g0", Halo(coreProfile, 9.086059744049174e+16, 7.7e-3), bulge_MM17, gas_HI_MM17, gas_H2_MM17, DoubleDiscStellarModel(stellar_mass_model_C03, 8.87e+14, 3.0e-4, 2.36e-3, 1.487e+14, 9.0e-4, 3.29e-3), 1e+4 * MYR_TO_S)



##################################################

cache_location::String = ".cache/"

function _save(host::HostModel, s::Symbol)
    
    r = 10.0.^range(log10(1e-3 * host.halo.rs), log10(host.rt), 100)

    @info "| Saving " * string(s) * " in cache" 

    y = Array{Float64}(undef, 100)
    Threads.@threads for ir in 1:100
        y[ir] = @eval $s($(Ref(r))[][$(Ref(ir))[]], $(Ref(host))[])
    end
    
    #y = @eval $s.($(Ref(r))[], $(Ref(host))[])
    
    JLD2.jldsave(cache_location * string(s)  * "_" * string(hash(host.name), base=16) * ".jld2" ; r = r, y = y)

    return true

end


## Possibility to interpolate the model
function _load(host::HostModel, s::Symbol)

    hash_value = hash(host.name)

    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = string(s) * "_" * string(hash_value, base=16) * ".jld2" 

    !(file in filenames) && _save(host, s)

    data    = JLD2.jldopen(cache_location * file)
    r = data["r"]
    y = data["y"]

    log10_y = Interpolations.interpolate((log10.(r),), log10.(y),  Interpolations.Gridded(Interpolations.Linear()))
    #return @eval $s(x::Real) = 10.0^$(Ref(log10_y))[](log10(x))

    function return_func(x::Real)
        
        if (x < r[1]) || (x >= r[end])
            return @eval $s($(Ref(x))[], $(Ref(host))[])
        end

        return 10.0^log10_y(log10(x))
    end

    return return_func
end

# preload a number of functions
function load!(host::HostModel)
    for field in fieldnames(HostModel)
        (getfield(host, field) === nothing) && setfield!(host, field, _load(host, field))
    end
end


end