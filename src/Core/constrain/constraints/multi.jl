#=
!NOTE: This is a convenience wrapper in case one wants to define a vector of parameter in a single Param struct.
Note that doing so will most likely result in worse performance, but more can be more convenient when testing models.
!NOTE: At the moment this also means that the arrays of Constraints need be of type <: AbstractConstraint (not just a valid distribution)
=#
############################################################################################
# 1. Create a new Constraint, MyConstraint <: AbstractConstraint.
struct MultiConstraint{D<:Union{Vector{C}, Vector{<:Array{C}}} where {C<:AbstractConstraint}} <: AbstractConstraint
    constraint::D
    function MultiConstraint(constraint::D) where {D<:Union{Vector{C}, Vector{<:Array{C}}} where {C<:AbstractConstraint}}
        return new{D}(constraint)
    end
end

############################################################################################
# Add method to directly state distribution for parameter in Params
Param(_rng::Random.AbstractRNG, constraint::D, val::B) where {D<:Vector{<:Distributions.Distribution},B} =
    Param(_rng, MultiConstraint(map(iter -> DistributionConstraint(constraint[iter]), eachindex(constraint))), val)
Param(_rng::Random.AbstractRNG, constraint::D, val::B) where {D<:Vector{<:Array{<:Distributions.Distribution}},B} =
    Param(_rng, MultiConstraint( map(iter -> DistributionConstraint.(constraint[iter]), eachindex(constraint)) ), val)

############################################################################################
#=
2.1 Define functions to unconstrain(constraint, val) to unconstrained domain valᵤ, and a function constrain(constraint, valᵤ) back to val.
Dimensions of val and valᵤ should be the same, flattening will be handled separately.
=#
function unconstrain(multi::MultiConstraint, val)
    return unconstrain(multi.constraint, val)
end
function constrain(multi::MultiConstraint, valᵤ)
    return constrain(multi.constraint, valᵤ)
end

############################################################################################
# 3. Optional - Check if check_transformer(constraint, val) works
#=
con3 = [Simplex(3), Simplex(3)]
val3 = [ [.3, .5, .2], [.9, .05, .05]]

con4 = [[CorrelationMatrix(), CorrelationMatrix()], [CorrelationMatrix(), CorrelationMatrix()]]
a = [1. .7 ; .7 1.0]
val4 = [[copy(a), copy(a)], [copy(a), copy(a)]]

con5 = [Normal(1.), Normal(1)]
val5 = [2., 3.]

con6 = [[Normal(1.), Normal(1)], [Normal(1.), Normal(1)]]
val6 = [[2., 3.], [2., 3.]]
=#

############################################################################################
# 4. If Objective is used, include a method that computes logabsdet from transformation to unconstrained domain. Same syntax as in Bijectors.jl package is used, i.e., -log_abs_det_jac is returned for computations.
function log_abs_det_jac(multi::MultiConstraint, θ::T) where {T}
    return log_abs_det_jac(multi.constraint, θ)
end

############################################################################################
# 5. Add _check function to check for all other peculiar things that should be tested if new releases come out and add to Test section.
function _check(_rng::Random.AbstractRNG, multi::MultiConstraint, val::V) where {V}
    return _check(_rng, multi.constraint, val)
end
function _check(_rng::Random.AbstractRNG, constraint::AbstractArray, val::V,) where {V}
    return all( map(iter -> _check(_rng, constraint[iter], val[iter]), eachindex(constraint)) )
end

function check_constraint(multi::MultiConstraint, val::V) where {V}
    return check_constraint(multi.constraint, val)
end
function check_constraint(constraint::AbstractArray, val::V) where {V}
    return all( map(iter -> check_constraint(constraint[iter], val[iter]), eachindex(constraint)) )
end

############################################################################################
# 6. Optionally - If flattening occurs, use Abstract Array nested structure to flatten based on each individual constraint.
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    multi::MultiConstraint,
    x
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    U<:UnflattenTypes
}
    return construct_flatten(T, flattentype, unflattentype, multi.constraint, x)
end

############################################################################################
# Additional functionality -- that is not considered for other AbstractConstraints -- to evaluate prior logdensities etc that make logposterior definitions easier.
function sample_constraint(_rng::Random.AbstractRNG, multi::MultiConstraint, val)
    return sample_constraint(_rng, multi.constraint, val)
end
function log_prior(multi::MultiConstraint, θ::T) where {T}
    return log_prior(multi.constraint, θ)
end
function log_prior_with_transform(multi::MultiConstraint, θ::T) where {T}
    return log_prior_with_transform(multi.constraint, θ)
end
function _checkprior(multi::MultiConstraint)
    return _checkprior(multi.constraint)
end
function _checksampleable(multi::MultiConstraint)
    return _checksampleable(multi.constraint)
end

############################################################################################
#Export
export
    MultiConstraint,
    construct_flatten,
    Param,
    constrain,
    unconstrain,
    log_abs_det_jac,
    sample_constraint,
    log_prior,
    log_prior_with_transform
