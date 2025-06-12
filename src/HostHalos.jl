module HostHalos

import CosmoTools: BkgCosmology, planck18_bkg, lookback_redshift, δt, z_to_a, Cosmology, planck18, MPC_TO_KM, MYR_TO_S, G_NEWTON
import CosmoTools: Halo, HaloProfile, nfwProfile, αβγProfile, m_halo, ρ_halo, μ_halo, coreProfile, rΔ, convert_Halo
import CosmoTools: constant_G_NEWTON, MegaParsecs, KiloMeters, Seconds, Msun, convert_lengths, GigaYears, convert_times
import CosmoTools: dflt_bkg_cosmo, dflt_cosmo, HaloType

import QuadGK, JLD2,  Interpolations, SpecialFunctions, Random

export ρ_HI, ρ_H2, ρ_ISM, ρ_baryons, ρ_baryons_spherical, host_halo, m_baryons_spherical
export m_host_spherical, μ_host_spherical, ρ_host_spherical, σ_baryons, age_host
export HostModel, BulgeModel, GasModel, StellarMassModel, StellarModel, HostModelType, HostInterpolation, HostInterpolationType
export circular_velocity, circular_period, number_circular_orbits, velocity_dispersion_spherical
export circular_velocity_kms, velocity_dispersion_spherical_kms 
export milky_way_MM17_g1, milky_way_MM17_g0
export maximum_impact_parameter, number_stellar_encounters, stellar_mass_function, stellar_mass_model_C03, moments_stellar_mass
export load!, make_cache
export convert_HostModel, get_host_halo_type

include("Model.jl")
include("Interpolation.jl")

# define the union of the two models
#Host{T} = Union{HostModel{T}, HostInterpolation{T}}

end