############################################################################################
# 1. Create a new Constraint, MyConstraint <: AbstractConstraint.
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter - keeps parameter unconstrained. Useful for assigning buffer values for functions of parameter.

# Fields
$(TYPEDFIELDS)
"""
struct Unconstrained{B<:Bijection} <: AbstractConstraint
    bijection::B
    function Unconstrained()
        bijection = Bijection(Bijectors.identity)
        return new{typeof(bijection)}(bijection)
    end
end

############################################################################################
#=
2.1 Define functions to unconstrain(constraint, val) to unconstrained domain valᵤ, and a function constrain(constraint, valᵤ) back to val.
Dimensions of val and valᵤ should be the same, flattening will be handled separately.
=#
function unconstrain(unconstrained::Unconstrained, val)
    return unconstrain(unconstrained.bijection, val)
end
function constrain(unconstrained::Unconstrained, valᵤ)
    return constrain(unconstrained.bijection, valᵤ)
end

############################################################################################
# 3. Optional - Check if check_transformer(constraint, val) works
#=
b = Bijectors.identity
binv = Bijectors.inverse(b)

constraint = Unconstrained()
val = 2.0
check_constraint(constraint, val)
val_u = unconstrain(constraint, val)
val_o = constrain(constraint, val_u)
=#

############################################################################################
# 4. If Objective is used, include a method that computes logabsdet from transformation to unconstrained domain. Same syntax as in Bijectors.jl package is used, i.e., -log_abs_det_jac is returned for computations.
function log_abs_det_jac(unconstrained::Unconstrained, θ::T) where {T}
    return log_abs_det_jac(unconstrained.bijection, θ)
end

############################################################################################
# 5. Add _check function to check for all other peculiar things that should be tested if new releases come out and add to Test section.
function _check(
    _rng::Random.AbstractRNG,
    constraint::Unconstrained,
    val::Union{R,Array{R},AbstractArray}
) where {R<:Real}
    return true
end

############################################################################################
#Export
export
    Unconstrained,
    constrain,
    unconstrain,
    log_abs_det_jac
