############################################################################################
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter - keeps scalar parameter constrained.

# Fields
$(TYPEDFIELDS)
"""
struct Constrained{B<:Bijection} <: AbstractConstraint
    bijection::B
    function Constrained(min::S, max::T, ϵ = 1e-6) where {S<:AbstractFloat, T<:AbstractFloat}
        ArgCheck.@argcheck min < max
        vals = promote(min, max)
        F = eltype(vals)
        b = Bijection(Bijectors.TruncatedBijector(F(vals[1] + ϵ), F(vals[2] - ϵ)))
        return new{typeof(b)}(b)
    end
end

############################################################################################
#=
2.1 Define functions to unconstrain(constraint, val) to unconstrained domain valᵤ, and a function constrain(constraint, valᵤ) back to val.
Dimensions of val and valᵤ should be the same, flattening will be handled separately.
=#
function unconstrain(constrained::Constrained, val)
    return unconstrain(constrained.bijection, val)
end
function constrain(constrained::Constrained, valᵤ)
    return constrain(constrained.bijection, valᵤ)
end

############################################################################################
# 3. Optional - Check if check_transformer(constraint, val) works
#=
constraint = Constrained(3., 10.)
val = 5.0
check_constraint(constraint, val)
val_u = unconstrain(constraint, val)
val_o = constrain(constraint, val_u)
=#

############################################################################################
# 4. If Objective is used, include a method that computes logabsdet from transformation to unconstrained domain. Same syntax as in Bijectors.jl package is used, i.e., -log_abs_det_jac is returned for computations.
function log_abs_det_jac(constrained::Constrained, θ::T) where {T}
    return log_abs_det_jac(constrained.bijection, θ)
end

############################################################################################
# 5. Add _check function to check for all other peculiar things that should be tested if new releases come out and add to Test section.
function _check(
    _rng::Random.AbstractRNG,
    constraint::Constrained,
    val::R,
) where {R<:Real}
    min = constraint.bijection.unconstrain.lb
    max = constraint.bijection.unconstrain.ub
#    ArgCheck.@argcheck typeof(val) == typeof(min) == typeof(max) "Type of Constrained boundaries must match type of val"
    ArgCheck.@argcheck min < val < max "val must be between boundaries"
    return true
end

############################################################################################
#Export
export
    Constrained,
    constrain,
    unconstrain,
    log_abs_det_jac
