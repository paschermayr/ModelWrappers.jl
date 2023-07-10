#!NOTE: Make Distribution.Distributions work in ModelWrappers such that it can be directly used as a constraint via Bijectors.
# In order to do so, we create a separate Constraint that includes a Bijection constraint.

############################################################################################
# 1. Create a new Constraint, MyConstraint <: AbstractConstraint.
struct DistributionConstraint{D<:Distributions.Distribution, B<:Bijection} <: AbstractConstraint
    dist::D
    bijection::B
    function DistributionConstraint(dist::D) where {D<:Distributions.Distribution}
        bijection = Bijection(Bijectors.bijector(dist))
        return new{D, typeof(bijection)}(dist, bijection)
    end
end

############################################################################################
# Add method to directly state distribution for parameter in Params, instead of manually defining DistributionConstraint(distribution)
Param(_rng::Random.AbstractRNG, constraint::A, val::B) where {A<:Distributions.Distribution,B} = Param(_rng, DistributionConstraint(constraint), val)
#Param(constraint::A, val::B) where {A<:Distributions.Distribution, B} = Param(Random.GLOBAL_RNG, DistributionConstraint(constraint), val)

############################################################################################
#=
2.1 Define functions to unconstrain(constraint, val) to unconstrained domain valᵤ, and a function constrain(constraint, valᵤ) back to val.
Dimensions of val and valᵤ should be the same, flattening will be handled separately.
=#
function unconstrain(dist::DistributionConstraint, val)
    return unconstrain(dist.bijection, val)
end
function constrain(dist::DistributionConstraint, valᵤ)
    return constrain(dist.bijection, valᵤ)
end

############################################################################################
# 3. Optional - Check if check_transformer(constraint, val) works
#=
constraint = DistributionConstraint(Gamma(2,2))
val = 2.0
check_constraint(constraint, val)
val_u = unconstrain(constraint, val)
val_o = constrain(constraint, val_u)
=#

############################################################################################
# 4. If Objective is used, include a method that computes logabsdet from transformation to unconstrained domain. Same syntax as in Bijectors.jl package is used, i.e., -log_abs_det_jac is returned for computations.
function log_abs_det_jac(dist::DistributionConstraint, θ::T) where {T}
    return log_abs_det_jac(dist.bijection, θ)
end

############################################################################################
# 5. Add _check function to check for all other peculiar things that should be tested if new releases come out and add to Test section.
function _check(
    _rng::Random.AbstractRNG,
    d::DistributionConstraint,
    val::Union{Factorization, R, Array{R}, AbstractArray},
) where {R<:Real}
    _val = rand(_rng, d.dist)
    return _check(_rng, d.bijection, val) && typeof(val) == typeof(_val) && size(val) == size(_val) ? true : false
end

############################################################################################
# 6.1 Optionally - Ignore non-specified distributions when flattening
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    constraint::Distributions.Distribution,
    x::Union{Factorization, R, Array{R}},
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    U<:UnflattenTypes,
    R<:Real,
}
    return construct_flatten(T, flattentype, unflattentype, x)
end

# 6.2 Implement custom flattening behavior for Bijectors

############################################################################################
# Additional functionality -- that is not considered for other AbstractConstraints -- to evaluate prior logdensities etc that make logposterior definitions easier.
function sample_constraint(_rng::Random.AbstractRNG, constraint::DistributionConstraint, val)
    return rand(_rng, constraint.dist)
end
function sample_constraint(_rng::Random.AbstractRNG, prior::Distributions.Distribution, val)
    return rand(_rng, prior)
end
#=
function sample_constraint(_rng::Random.AbstractRNG, priorᵥ::Vector{<:Distributions.Distribution}, val::AbstractVector)
    return rand.(_rng, priorᵥ)
end
=#
function log_prior(constraint::DistributionConstraint, θ::T) where {T}
    return log_prior(constraint.dist, θ)
end
function log_prior(prior::Distributions.Distribution, θ::T) where {T}
    return Distributions.logpdf(prior, θ)
end
#=
function log_prior(priorᵥ::Vector{<:Distributions.Distribution}, θ::AbstractVector)
    return sum(Distributions.logpdf.(priorᵥ, θ))
end
=#
function log_prior_with_transform(constraint::DistributionConstraint, θ::T) where {T}
    return log_prior_with_transform(constraint.dist, θ)
end
function log_prior_with_transform(prior::Distributions.Distribution, θ::T) where {T}
    return Bijectors.logpdf_with_trans(prior, θ, true)
end
#=
function log_prior_with_transform(
    priorᵥ::Vector{<:Distributions.Distribution}, θ::AbstractVector
)
    return sum(Bijectors.logpdf_with_trans.(priorᵥ, θ, true))
end
=#
function _checkprior(constraint::S) where {S<:DistributionConstraint}
    return true
end
function _checksampleable(constraint::S) where {S<:DistributionConstraint}
    return true
end

############################################################################################
#Export
export
    DistributionConstraint,
    Param,
    constrain,
    unconstrain,
    log_abs_det_jac,
    sample_constraint,
    log_prior,
    log_prior_with_transform
