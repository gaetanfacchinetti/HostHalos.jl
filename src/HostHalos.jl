module HostHalos

import CosmoTools: BkgCosmology, planck18_bkg, lookback_redshift, δt_s, z_to_a, Cosmology, planck18, MPC_TO_KM, MYR_TO_S, G_NEWTON
import CosmoTools: Halo, HaloProfile, nfwProfile, αβγProfile, m_halo, ρ_halo, μ_halo, coreProfile, rΔ

import QuadGK, JLD2,  Interpolations

export ρ_HI, ρ_H2, ρ_ISM, ρ_baryons, ρ_baryons_spherical, host_halo, m_baryons_spherical
export m_host_spherical, μ_host_spherical, ρ_host_spherical, σ_baryons, age_host
export HostModel, BulgeModel
export circular_velocity, circular_period, number_circular_orbits, velocity_dispersion_spherical
export circular_velocity_kms, velocity_dispersion_spherical_kms 
export milky_way_MM17_g1, milky_way_MM17_g0


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

    return HostModel(name, halo, bulge, gas_HI, gas_H2, stars, age, rt, nothing, nothing, nothing, nothing, nothing)
end

function Base.getproperty(obj::HostModel, s::Symbol)

    # we load the data if necessary
    if getfield(obj, s) === nothing
        setfield!(obj, s, _load(obj, s))
    end

    return getfield(obj, s)
end

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


""" circular velocity in (Mpc / s) for `r` in Mpc """
circular_velocity(r::Real, host::HostModel = milky_way_MM17_g1) = sqrt(G_NEWTON * m_host_spherical(r, host) / r) 

""" circular velocity in (km / s) for `r` in Mpc """
circular_velocity_kms(r::Real, host::HostModel = milky_way_MM17_g1) = circular_velocity(r, host) *  MPC_TO_KM

""" circular period in s for `r` in Mpc """
circular_period(r::Real, host::HostModel = milky_way_MM17_g1) = 2.0 * π * r / circular_velocity(r, host)  

""" number or circular orbits with `r` in Mpc """
number_circular_orbits(r::Real, host::HostModel = milky_way_MM17_g1, z::Real = 0, cosmo::BkgCosmology = planck18_bkg; kws...) = floor(Int, age_host(z, host, cosmo, kws...) / circular_period(r, host))

""" Jeans dispersion in (Mpc / s)"""
velocity_dispersion_spherical(r::Real, host::HostModel = milky_way_MM17_g1) = (r >= host.rt) ? 0.0 : sqrt(G_NEWTON / ρ_halo(r, host.halo) * QuadGK.quadgk(rp -> ρ_halo(rp, host.halo) * m_host_spherical(rp, host)/rp^2, r, host.rt, rtol=1e-3)[1])

""" Jeans dispersion in (km / s)"""
velocity_dispersion_spherical_kms(r::Real, host::HostModel = milky_way_MM17_g1) = velocity_dispersion_spherical(r, host) * MPC_TO_KM



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
    thin_σ0::T
    thin_zd::T
    thin_rd::T
    thick_σ0::T
    thick_zd::T
    thick_rd::T  
end

DoubleDiscStellarModel(thin_σ0::Real, thin_rd::Real, thin_zd::Real, thick_σ0::Real, thick_rd::Real, thick_zd::Real) = DoubleDiscStellarModel(promote(thin_σ0, thin_rd, thin_zd, thick_σ0, thick_rd, thick_zd)...)
ρ_stars(r::Real, z::Real, stars::DoubleDiscStellarModel) = ρ_exponential_disc(r, z, σ0 = stars.thick_σ0, zd = stars.thick_zd, rd = stars.thick_rd) + ρ_exponential_disc(r, z, σ0 = stars.thin_σ0, zd = stars.thin_zd, rd = stars.thin_rd)
σ_stars(r::Real, stars::DoubleDiscStellarModel) = σ_exponential_disc(r, σ0 = stars.thick_σ0, rd = stars.thick_rd) + σ_exponential_disc(r, σ0 = stars.thin_σ0, rd = stars.thin_rd)

const bulge_MM17::AxiSymmetricBGBulgeModel = AxiSymmetricBGBulgeModel(9.73e+19, 7.5e-5, 0.5, 1.8, 2.1e-3)
const gas_HI_MM17::SechGasModel = SechGasModel(5.31e+13, 8.5e-5, 4.0e-3, 7.0e-3)
const gas_H2_MM17::SechGasModel = SechGasModel(2.180e+15, 4.5e-5, 1.2e-2, 1.5e-3) 

milky_way_MM17_g1 = HostModel("MilkyWay_MM17_g1", Halo(nfwProfile, 9.24412866426226e+15, 1.86e-2),  bulge_MM17, gas_HI_MM17, gas_H2_MM17, DoubleDiscStellarModel(8.87e+14, 3.0e-4, 2.53e-3, 1.487e+14, 9.0e-4, 3.29e-3), 1e+4 * MYR_TO_S)
milky_way_MM17_g0 = HostModel("MilkyWay_MM17_g0", Halo(coreProfile, 9.086059744049174e+16, 7.7e-3), bulge_MM17, gas_HI_MM17, gas_H2_MM17, DoubleDiscStellarModel(8.87e+14, 3.0e-4, 2.36e-3, 1.487e+14, 9.0e-4, 3.29e-3), 1e+4 * MYR_TO_S)


#########################

cache_location::String = ".cache/"

function _save(host::HostModel, s::Symbol)
    
    r = 10.0.^range(log10(1e-3 * host.halo.rs), log10(host.rt), 100)

    @info "| Saving " * string(s) * " in cache" 
    y = @eval $s.($(Ref(r))[], $(Ref(host))[])
    
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

    return x-> 10.0^log10_y(log10(x))
end


end