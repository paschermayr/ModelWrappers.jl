############################################################################################
# 1. Create a new Constraint, MyConstraint <: AbstractConstraint.
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter - keeps parameter fixed. Useful for assigning buffer values for functions of parameter.

# Fields
$(TYPEDFIELDS)
"""
struct Fixed{B<:Bijection} <: AbstractConstraint
    bijection::B
    function Fixed()
        bijection = Bijection(Bijectors.identity)
        return new{typeof(bijection)}(bijection)
    end
end

############################################################################################
#=
2. Define functions to unconstrain(constraint, val) to unconstrained domain valᵤ, and a function constrain(constraint, valᵤ) back to val.
Dimensions of val and valᵤ should be the same, flattening will be handled separately.
=#
function unconstrain(fixed::Fixed, val)
    return unconstrain(fixed.bijection, val)
end
function constrain(fixed::Fixed, valᵤ)
    return constrain(fixed.bijection, valᵤ)
end

############################################################################################
# 3. Optional - Check if check_transformer(constraint, val) works
#=
b = Bijectors.identity
binv = Bijectors.inverse(b)

constraint = Fixed()
val = 2.0
check_constraint(constraint, val)
val_u = unconstrain(constraint, val)
val_o = constrain(constraint, val_u)
=#

############################################################################################
# 4. If Objective is used, include a method that computes logabsdet from transformation to unconstrained domain. Same syntax as in Bijectors.jl package is used, i.e., -log_abs_det_jac is returned for computations.
function log_abs_det_jac(fixed::Fixed, θ::T) where {T}
    return log_abs_det_jac(fixed.bijection, θ)
end

############################################################################################
# 5. Add _check function to check for all other peculiar things that should be tested if new releases come out and add to Test section.
function _check(
    _rng::Random.AbstractRNG,
    constraint::Fixed,
    val::Union{R,Array{R},AbstractArray},
) where {R<:Real}
    return true
end
function check_constraint(constraint::Fixed, val::V) where {V}
    return true
end

############################################################################################
# 6. Optionally - choose to only flatten k-1 parameter if Simplex is constraint
function _construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    x
) where {T<:Real,F<:FlattenTypes,U<:UnflattenTypes}
    v = T[]
    _flatten_Fixed(x) = v
    _unflatten_Fixed(v) = x
    return _flatten_Fixed, _unflatten_Fixed
end
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    constraint::Fixed,
    x,
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    return _construct_flatten(T, flattentype, unflattentype, x)
end

############################################################################################
#Export
export
    Fixed,
    construct_flatten,
    constrain,
    unconstrain,
    log_abs_det_jac
