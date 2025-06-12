
abstract type BulgeModel{T<:AbstractFloat} end
abstract type GasModel{T<:AbstractFloat} end
abstract type StellarMassModel{T<:AbstractFloat} end
abstract type StellarModel{T<:AbstractFloat, SMM<:StellarMassModel{T}} end


struct HostModel{
    T<:AbstractFloat, 
    P<:HaloProfile,
    BM<:BulgeModel{T}, 
    GMHI<:GasModel{T}, 
    GMH2<:GasModel{T}, 
    SM<:StellarModel{T, <:StellarMassModel{T}}}

    name::String
    halo::Halo{T, P}
    bulge::BM
    gas_HI::GMHI
    gas_H2::GMH2
    stars::SM

    age::T # in Gyrs
    rt::T # in Mpc

end

const HostModelType{T} = HostModel{
    T,
    <:HaloProfile,
    <:BulgeModel{T},
    <:GasModel{T},
    <:GasModel{T},
    <:StellarModel{T, <:StellarMassModel{T}}
}

Base.length(::HostModel) = 1
Base.iterate(iter::HostModel) = (iter, nothing)
Base.iterate(::HostModel, state::Nothing) = nothing

get_host_halo_type(::HostModelType{T}) where {T<:AbstractFloat} = T

function convert_HostModel(::Type{T}, model::HostModel) where {T<:AbstractFloat}
    
    return HostModel(
        model.name,
        convert_Halo(T, model.halo),
        convert_BulgeModel(T, model.bulge),
        convert_GasModel(T, model.gas_HI),
        convert_GasModel(T, model.gas_H2),
        convert_StellarModel(T, model.stars),
        convert(T, model.age),
        convert(T, model.rt),
    )

end


function HostModel(
    name::String, 
    halo::HaloType{T}, 
    bulge::BulgeModel{T}, 
    gas_HI::GasModel{T}, 
    gas_H2::GasModel{T}, 
    stars::StellarModel{T}, 
    age::T
    ) where {T<:AbstractFloat}

    # by default the tidal radius is set to the virial radius
    rt = rΔ(halo)

    return HostModel(name, halo, bulge, gas_HI, gas_H2, stars, age, rt)
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


ρ_HI(r::T, z::T, host::HostModelType{T}) where {T<:AbstractFloat} = ρ_gas(r, z, host.gas_HI)
ρ_H2(r::T, z::T, host::HostModelType{T}) where {T<:AbstractFloat} = ρ_gas(r, z, host.gas_H2)
ρ_ISM(r::T, z::T, host::HostModelType{T}) where {T<:AbstractFloat} = ρ_HI(r, z, host) + ρ_H2(r, z, host) 
ρ_bulge(r::T, z::T, host::HostModelType{T}) where {T<:AbstractFloat} = ρ_bulge(r, z, host.bulge)
ρ_stars(r::T, z::T, host::HostModelType{T}) where {T<:AbstractFloat} = ρ_stars(r, z, host.stars)
ρ_baryons(r::T, z::T, host::HostModelType{T}) where {T<:AbstractFloat} = ρ_ISM(r, z, host) + ρ_bulge(r, z, host) + ρ_stars(r, z, host) 

# Surface density (for disc galaxies)
σ_HI(r::T, host::HostModelType{T}) where {T<:AbstractFloat} = σ_gas(r, host.gas_HI)
σ_H2(r::T, host::HostModelType{T}) where {T<:AbstractFloat} = σ_gas(r, host.gas_H2)
σ_ISM(r::T, host::HostModelType{T}) where {T<:AbstractFloat} = σ_HI(r, host) + σ_H2(r, host)
σ_bulge(r::T, host::HostModelType{T}) where {T<:AbstractFloat} = σ_bulge(r, host.bulge)
σ_stars(r::T, host::HostModelType{T}) where {T<:AbstractFloat} = σ_stars(r, host.stars)
σ_baryons(r::T, host::HostModelType{T}) where {T<:AbstractFloat} = σ_ISM(r, host) + σ_bulge(r, host) + σ_stars(r, host) 



""" age of the host in Gyrs at any redshift """
function age_host(z::T, host::HostModelType{T}, cosmo::BkgCosmology{T} = dflt_bkg_cosmo(T); kws...) where {T<:AbstractFloat}
    z_max = lookback_redshift(host.age)
    @assert z_max > z "z must be below the formation redshift of the galaxy, z_max = $z_max"
    return δt(z_to_a(z_max), z_to_a(z), cosmo; kws...)
end


###########################
## SPHERICISED QUANTITIES

function der_ρ_baryons_spherical(xp::T, r::T, host::HostModelType{T})::T where {T<:AbstractFloat}

    y = sqrt(1 - xp^2)
    return xp^2/y * (ρ_baryons(r * xp, - r * y, host) + ρ_baryons(r * xp, r * y, host))/T(2)

end

function ρ_baryons_spherical(r::T, host::HostModelType{T}) where {T<:AbstractFloat}  
    _to_integrate(xp::T)::T  = der_ρ_baryons_spherical(xp, r, host)
    return QuadGK.quadgk(_to_integrate, T(0), T(1), rtol=T(1e-4))[1] 
end

function m_baryons_spherical(r::T, host::HostModelType{T}) where {T<:AbstractFloat} 
    _to_integrate(rp::T)::T = rp^2 * ρ_baryons_spherical(rp, host)
    return T(4 * π) * QuadGK.quadgk(_to_integrate, T(0), r, rtol=T(1e-3))[1] 
end

ρ_host_spherical(r::T, host::HostModelType{T}) where {T<:AbstractFloat} = ρ_baryons_spherical(r, host) + ρ_halo(r, host.halo)
m_host_spherical(r::T, host::HostModelType{T}) where {T<:AbstractFloat} = m_baryons_spherical(r, host) + m_halo(r, host.halo)

mΔ(h::HaloType{T}, Δ::T = T(200), cosmo::C = dflt_cosmo(T)) where {T<:AbstractFloat, C<:Cosmology{T, <:BkgCosmology{T}}}  = mΔ(h, Δ, cosmo.bkg.ρ_c0)

""" circular velocity in (Mpc / Gyrs) for `r` in Mpc """
circular_velocity(r::T, host::HostModelType{T} = milky_way_MM17_g1) where {T<:AbstractFloat} = sqrt(constant_G_NEWTON(MegaParsecs, Msun, GigaYears, T) * m_host_spherical(r, host)/ r) 

""" circular velocity in (km / s) for `r` in Mpc """
circular_velocity_kms(r::T, host::HostModelType{T} = milky_way_MM17_g1) where {T<:AbstractFloat}  = sqrt(constant_G_NEWTON(KiloMeters, Msun, Seconds, T) * m_host_spherical(r, host) / convert_lengths(r, MegaParsecs, KiloMeters)) 
 
""" circular period in Gyrs for `r` in Mpc """
circular_period(r::T, host::HostModelType{T} = milky_way_MM17_g1)  where {T<:AbstractFloat} = T(2) * π * r / circular_velocity(r, host)  

""" number or circular orbits with `r` in Mpc """
number_circular_orbits(r::T, host::HostModelType{T} = milky_way_MM17_g1, z::T = T(0), bkg_cosmo::BKG = dflt_bkg_cosmo(T); kws...)  where {T<:AbstractFloat, BKG<:BkgCosmology{T}}  = floor(Int, age_host(z, host, bkg_cosmo, kws...) / circular_period(r, host))


struct IntegrandVelocityDisp{
    T<:AbstractFloat, 
    P<:HaloProfile,
    BM<:BulgeModel{T}, 
    GMHI<:GasModel{T}, 
    GMH2<:GasModel{T}, 
    SM<:StellarModel{T, <:StellarMassModel{T}}
    }

    host::HostModel{T, P, BM, GMHI, GMH2, SM}
end

function (f::IntegrandVelocityDisp{T, P, BM, GMHI, GMH2, SM})(rp::T)::T where {
    T<:AbstractFloat,
    P<:HaloProfile,
    BM<:BulgeModel{T},
    GMHI<:GasModel{T},
    GMH2<:GasModel{T},
    SM<:StellarModel{T, <:StellarMassModel{T}}}

    return ρ_halo(rp, f.host.halo) * m_host_spherical(rp, f.host) / (rp^2)
end

""" Jeans dispersion in (Mpc / s) """
function velocity_dispersion_spherical(r::T, host::HostModelType{T} = milky_way_MM17_g1) where {T<:AbstractFloat} 

    f = IntegrandVelocityDisp(host)

    if (r >= host.rt)
        return T(0) 
    else
        # need to put the T outside because Float32(G_NEWTON) = 0.0f0
        return T(sqrt(G_NEWTON / ρ_halo(r, host.halo) * QuadGK.quadgk(f, r, host.rt, rtol=T(1e-3))[1]))   
    end

end 

""" Jeans dispersion in (km / s) """
velocity_dispersion_spherical_kms(r::T, host::HostModelType{T} = milky_way_MM17_g1) where {T<:AbstractFloat}  = velocity_dispersion_spherical(r, host) * T(MPC_TO_KM)





##################################################
# Stellar mass and relative velocity distributions
function stellar_mass_function_LNPL(m::T, m_cut::NTuple{4, T}, norm::NTuple{4, T}, indices::NTuple{3, T}, med::T, σm::T) where {T<:AbstractFloat}
    
    log10_m = log10(m)

    (log10_m <= m_cut[1]) && (return norm[1] * exp(-(log10_m - log10(med))^2 / (T(2) * (σm)^2)) / m) 
    (m_cut[1] < log10_m && log10_m <= m_cut[2]) && (return norm[2] *  m^indices[1])
    (m_cut[2] < log10_m && log10_m <= m_cut[3]) && (return norm[3] *  m^indices[2])
    (m_cut[3] < log10_m && log10_m <= m_cut[4]) && (return norm[4] *  m^indices[3])

    return T(0)
end


cdf_LNPL(m0::T, m1::T, index::T) where {T<:AbstractFloat} = index != -1 ? (m1^(1+index) - m0^(1+index))/(1+index) : log(m1/m0) 

function stellar_cdf_LNPL(m::T, m_cut::NTuple{4, T}, norm::NTuple{4, T}, indices::NTuple{3, T}, med::T, σm::T) where {T<:AbstractFloat}

    log10_m = log10(m)

    res =  norm[1] * sqrt(π/2) * σm * (1 + SpecialFunctions.erf((min(log10_m, m_cut[1]) - log10(med))/(sqrt(2) * σm)) ) * log(10)

    log10_m > m_cut[1] && (res += norm[2] * cdf_LNPL( exp10(m_cut[1]), min(exp10(m_cut[2]), m), indices[1]))
    log10_m > m_cut[2] && (res += norm[3] * cdf_LNPL( exp10(m_cut[2]), min(exp10(m_cut[3]), m), indices[2]))
    log10_m > m_cut[3] && (res += norm[4] * cdf_LNPL( exp10(m_cut[3]), min(exp10(m_cut[4]), m), indices[3]))

    return res

end


inv_cdf_LNPL(y::T, m0::T, norm::T, index::T, weight::T) where {T<:AbstractFloat} = index != -1 ?  ( (1+index) / norm * (y-weight)  + m0^(1+index) )^(1/(1+index)) : m0 * exp(y-weight)


function stellar_inv_cdf_LNPL(y::T, m_cut::NTuple{4, T}, norm::NTuple{4, T}, indices::NTuple{3, T}, weights::NTuple{4, T}, med::T, σm::T) where {T<:AbstractFloat}
    
    if y <= weights[1]
       return med * exp10( sqrt(2) * σm * SpecialFunctions.erfinv(sqrt(2/π) * y / norm[1] / σm / log(10) -1))
    end

    for i in 1:3
        (y > weights[i] && y <= weights[i+1]) && (return inv_cdf_LNPL(y, exp10(m_cut[i]), norm[i+1], indices[i], weights[i]))
    end

    throw(ArgumentError("Impossible to evaluate inv_cdf_LNPL for y = $y"))

end



function stellar_inv_cdf_LNPL(y::Vector{T}, m_cut::NTuple{4, T}, norm::NTuple{4, T}, indices::NTuple{3, T}, weights::NTuple{4, T}, med::T, σm::T) where {T<:AbstractFloat}
    
    res = Vector{T}(undef, length(y)) 

    mask = y .<= weights[1]
    res[mask] = med * exp10.( sqrt(2) * σm * SpecialFunctions.erfinv.(sqrt(2/π) * y[mask] / norm[1] / σm / log(10) .- 1))

    for i in 1:3
        mask = (y .> weights[i] .&& y .<= weights[i+1])
        res[mask] = inv_cdf_LNPL.(y[mask], exp10(m_cut[i]), norm[i+1], indices[i], weights[i])
    end

    return res

end




@inline function moments_stellar_mass_LNPL(n::Int, m_cut::NTuple{4, T}, norm::NTuple{4, T}, indices::NTuple{3, T}, med::T, σm::T) where {T<:AbstractFloat} 
    return QuadGK.quadgk(lnm -> exp(lnm)^(n+1) * stellar_mass_function_LNPL(exp(lnm), m_cut, norm, indices, med, σm), log(T(1e-7)), log(T(10.0^1.8)), rtol=T(1e-8))[1]  
end



""" Log-Lormal and Power-Law Mass model"""
struct LNPLStellarMassModel{T<:AbstractFloat} <: StellarMassModel{T}
    
    m_cut::NTuple{4, T} # in Msun
    norm::NTuple{4, T}
    indices::NTuple{3, T}
    med::T
    σm::T

    # computed quantity
    average_mstar::T   # in Msun
    average_mstar2::T  # in Msun^2
    weights::NTuple{4, T}

end


Base.length(::StellarMassModel) = 1
Base.iterate(iter::StellarMassModel) = (iter, nothing)
Base.iterate(::StellarMassModel, state::Nothing) = nothing

function convert_StellarMassModel(::Type{T}, model::LNPLStellarMassModel) where {T<:AbstractFloat} 

    return LNPLStellarMassModel(
        Tuple(T.(model.m_cut)), 
        Tuple(T.(model.norm)),
        Tuple(T.(model.indices)),
        convert.(T, getfield.(model, fieldnames(LNPLStellarMassModel)[4:end-1]))...,
        Tuple(T.(model.weights)))
end

function LNPLStellarMassModel(m_cut::NTuple{4, T}, norm::NTuple{4, T}, indices::NTuple{3, T}, med::T, σm::T) where {T<:AbstractFloat}
    
    # here we precompte the first and second moment of the stellar mass function if they are not given in input
    _av_mstar = moments_stellar_mass_LNPL(1, m_cut, norm, indices, med, σm)
    _av_mstar2 = moments_stellar_mass_LNPL(2, m_cut, norm, indices, med, σm)

    _weights = [stellar_cdf_LNPL(exp10(m_c), m_cut, norm, indices, med, σm) for m_c in m_cut]
    _weights[end] = T(1) # set the last weight to exactly 1
    
    return LNPLStellarMassModel(m_cut, norm, indices, med, σm, _av_mstar, _av_mstar2, Tuple(_weights))
end

const stellar_mass_model_C03::LNPLStellarMassModel{Float64} = LNPLStellarMassModel((0.0, 0.54, 1.26, 1.80), (0.2613, 0.0728, 0.02481, 0.0004135), (-5.37, -4.53, -3.11), 0.079, 0.69)#, 0.16794677064645963, 0.08941378784419536)

stellar_mass_function(m::T, model::SM = stellar_mass_model_C03) where {T<:AbstractFloat, SM<:LNPLStellarMassModel{T}}= stellar_mass_function_LNPL(m, model.m_cut, model.norm, model.indices, model.med, model.σm)
stellar_cdf(m::T, model::SM = stellar_mass_model_C03) where {T<:AbstractFloat, SM<:LNPLStellarMassModel{T}} = stellar_cdf_LNPL(m, model.m_cut, model.norm, model.indices, model.med, model.σm)
stellar_inv_cdf(y::T, model::SM = stellar_mass_model_C03) where {T<:AbstractFloat, SM<:LNPLStellarMassModel{T}} = stellar_inv_cdf_LNPL(y, model.m_cut, model.norm, model.indices, model.weights, model.med, model.σm)
stellar_inv_cdf(y::Vector{T}, model::SM = stellar_mass_model_C03) where {T<:AbstractFloat, SM<:LNPLStellarMassModel{T}} = stellar_inv_cdf_LNPL(y, model.m_cut, model.norm, model.indices, model.weights, model.med, model.σm)
moments_stellar_mass(n::Int, model::LNPLStellarMassModel{<:AbstractFloat}= stellar_mass_model_C03) =  moments_stellar_mass_LNPL(n, model.m_cut, model.norm, model.indices, model.med, model.σm)

stellar_mass_function(m::T, host::HostModelType{T}) where {T<:AbstractFloat} = stellar_mass_function(m, host.stars.mass_model)
stellar_cdf(m::T, host::HostModelType{T}) where {T<:AbstractFloat} = stellar_cdf(m, host.stars.mass_model)
stellar_inv_cdf(y::T, host::HostModelType{T}) where {T<:AbstractFloat} = stellar_inv_cdf(y, host.stars.mass_model)
stellar_inv_cdf(y::Vector{T}, host::HostModelType{T}) where {T<:AbstractFloat} = stellar_inv_cdf(y, host.stars.mass_model)
moments_stellar_mass(n::Int, host::HostModelType{T})  where {T<:AbstractFloat} = moments_stellar_mass(n, host.stars.mass_model)


xval(x::T, log10_med::T, σm::T) where {T<:AbstractFloat} = exp10(muladd(σm, x, log10_med))
function rand_stellar_mass_LNPL(n::Int, model::LNPLStellarMassModel{T}, rng::Random.AbstractRNG = Random.default_rng()) where {T<:AbstractFloat} 
    
    # allocate a vector for the output
    # exact lognormal distribution
    res = xval.(randn(rng, T, n), log10(model.med), model.σm)

    # mask and draw at the smaller values, which can be drawn from a gaussian (faster)
    mask = (res .> model.m_cut[1])

    # draw the higher values according to the inverse of the cdf
    res[mask] = stellar_inv_cdf_LNPL(rand(rng, T, count(mask)), model.m_cut, model.norm, model.indices, model.weights, model.med, model.σm)
    
    return res

end




""" maximum impact parameters at distance r (Mpc) from the Galactic center"""
function maximum_impact_parameter(r::T, host::HostModelType{T}) where {T<:AbstractFloat}
    return T(host.stars.mass_model.average_mstar^(1/3))/σ_stars(r, host) * T(QuadGK.quadgk(lnz -> exp(lnz) * ρ_stars(r, exp(lnz), host)^(2/3), log(T(1e-10)), log(T(1e+0)), rtol=T(1e-7))[1])
end

""" number of stars encountered on one disk crossing """
function number_stellar_encounters(r::T, host::HostModelType{T}, θ::T = T(π/3.0)) where {T<:AbstractFloat}
    return floor(Int, σ_stars(r, host) / host.stars.mass_model.average_mstar * π / cos(θ) * maximum_impact_parameter(r, host)^2)
end


###########################
## McMillan 2017 Model


# Definition of the profiles
function ρ_spherical_BG02(r::T, z::T; ρ0b::T, r0::T, q::T, α::T, rcut::T) where {T<:AbstractFloat}
    rp = sqrt(r^2 + (z/q)^2)
    return ρ0b/((1+rp/r0)^α) * exp(-(rp/rcut)^2)
end

ρ_exponential_disc(r::T, z::T; σ0::T, rd::T, zd::T)  where {T<:AbstractFloat} = σ0/(T(2)*zd)*exp(-abs(z)/zd - r/rd)
σ_exponential_disc(r::T; σ0::T, rd::T)  where {T<:AbstractFloat} = σ0 * exp(-r/rd)
ρ_sech_disc(r::T, z::T; σ0::T, rd::T, rm::T, zd::T)  where {T<:AbstractFloat} = σ0/(T(4)*zd)*exp(-rm/r - r/rd)*(sech(z/(T(2)*zd)))^2
σ_sech_disc(r::T; σ0::T, rd::T, rm::T, zd::T) where {T<:AbstractFloat} = σ0 * exp(-r/rd - rm/r)

struct AxiSymmetricBGBulgeModel{T<:AbstractFloat} <: BulgeModel{T}
    ρ0b::T
    r0::T
    q::T
    α::T
    rcut::T
end

Base.length(::BulgeModel) = 1
Base.iterate(iter::BulgeModel) = (iter, nothing)
Base.iterate(::BulgeModel, state::Nothing) = nothing

convert_BulgeModel(::Type{T}, model::AxiSymmetricBGBulgeModel) where {T<:AbstractFloat} = AxiSymmetricBGBulgeModel(convert.(T, getfield.(model, fieldnames(AxiSymmetricBGBulgeModel)))...)

ρ_bulge(r::T, z::T, bulge::AxiSymmetricBGBulgeModel{T}) where {T<:AbstractFloat} = ρ_spherical_BG02(r, z, ρ0b = bulge.ρ0b, r0 = bulge.r0, q = bulge.q, α = bulge.α, rcut = bulge.rcut)
σ_bulge(r::T, bulge::AxiSymmetricBGBulgeModel{T}) where {T<:AbstractFloat} = QuadGK.quadgk( z-> ρ_bulge(r, z, bulge), -T(10)*bulge.q/r, T(10)*bulge.q/r, rtol=T(1e-3))[1]

struct SechGasModel{T<:AbstractFloat} <: GasModel{T}
    σ0::T
    zd::T
    rm::T
    rd::T
end

Base.length(::GasModel) = 1
Base.iterate(iter::GasModel) = (iter, nothing)
Base.iterate(::GasModel, state::Nothing) = nothing

convert_GasModel(::Type{T}, model::SechGasModel) where {T<:AbstractFloat} = SechGasModel(convert.(T, getfield.(model, fieldnames(SechGasModel)))...)

ρ_gas(r::T, z::T, gas::SechGasModel{T}) where {T<:AbstractFloat}  = ρ_sech_disc(r, z, σ0 = gas.σ0, rd = gas.rd, rm = gas.rm, zd = gas.zd)
σ_gas(r::T, gas::SechGasModel{T})  where {T<:AbstractFloat} = σ_sech_disc(r, σ0 = gas.σ0, rd = gas.rd, rm = gas.rm, zd = gas.zd)

struct DoubleDiscStellarModel{T<:AbstractFloat, SMM<:StellarMassModel{T}} <: StellarModel{T, StellarMassModel{T}}
    mass_model::SMM
    thin_σ0::T
    thin_zd::T
    thin_rd::T
    thick_σ0::T
    thick_zd::T
    thick_rd::T  
end

Base.length(::StellarModel) = 1
Base.iterate(iter::StellarModel) = (iter, nothing)
Base.iterate(::StellarModel, state::Nothing) = nothing

function convert_StellarModel(::Type{T}, model::DoubleDiscStellarModel) where {T<:AbstractFloat} 
    
    return DoubleDiscStellarModel(
        convert_StellarMassModel(T, model.mass_model),
        convert.(T, getfield.(model, fieldnames(DoubleDiscStellarModel)[2:end]))...
        )
end

ρ_stars(r::T, z::T, stars::DoubleDiscStellarModel{T}) where {T<:AbstractFloat} = ρ_exponential_disc(r, z, σ0 = stars.thick_σ0, zd = stars.thick_zd, rd = stars.thick_rd) + ρ_exponential_disc(r, z, σ0 = stars.thin_σ0, zd = stars.thin_zd, rd = stars.thin_rd)
σ_stars(r::T, stars::DoubleDiscStellarModel{T}) where {T<:AbstractFloat} = σ_exponential_disc(r, σ0 = stars.thick_σ0, rd = stars.thick_rd) + σ_exponential_disc(r, σ0 = stars.thin_σ0, rd = stars.thin_rd)

const bulge_MM17::AxiSymmetricBGBulgeModel = AxiSymmetricBGBulgeModel(9.73e+19, 7.5e-5, 0.5, 1.8, 2.1e-3)
const gas_HI_MM17::SechGasModel = SechGasModel(5.31e+13, 8.5e-5, 4.0e-3, 7.0e-3)
const gas_H2_MM17::SechGasModel = SechGasModel(2.180e+15, 4.5e-5, 1.2e-2, 1.5e-3) 

milky_way_MM17_g1 = HostModel("MilkyWay_MM17_g1", Halo(nfwProfile, 9.24412866426226e+15, 1.86e-2),  bulge_MM17, gas_HI_MM17, gas_H2_MM17, DoubleDiscStellarModel(stellar_mass_model_C03, 8.87e+14, 3.0e-4, 2.53e-3, 1.487e+14, 9.0e-4, 3.29e-3), 10.0)
milky_way_MM17_g0 = HostModel("MilkyWay_MM17_g0", Halo(coreProfile, 9.086059744049174e+16, 7.7e-3), bulge_MM17, gas_HI_MM17, gas_H2_MM17, DoubleDiscStellarModel(stellar_mass_model_C03, 8.87e+14, 3.0e-4, 2.36e-3, 1.487e+14, 9.0e-4, 3.29e-3), 10.0)

