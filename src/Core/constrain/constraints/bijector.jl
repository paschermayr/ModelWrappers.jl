############################################################################################
# 1. Create a new Constraint, MyConstraint <: AbstractConstraint.
struct Bijection{U, C} <: AbstractConstraint
    unconstrain::U
    constrain::C
    function Bijection(bijector::F) where {F}
        #!NOTE: bijector used as input is used to transform val into unconstrained domain
        unconstrain = bijector
        constrain = Bijectors.inverse(unconstrain)
        return new{typeof(unconstrain), typeof(constrain)}(unconstrain, constrain)
    end
end

############################################################################################
#=
2. Define functions to unconstrain(constraint, val) to unconstrained domain valᵤ, and a function constrain(constraint, valᵤ) back to val.
Dimensions of val and valᵤ should be the same, flattening will be handled separately.
=#
function unconstrain(bijection::Bijection, val)
    valᵤ = Bijectors.transform(bijection.unconstrain, val)
    return valᵤ
end
function constrain(bijection::Bijection, valᵤ)
    val = Bijectors.transform(bijection.constrain, valᵤ)
    return val
end

############################################################################################
# 3. Optional - Check if check_transformer(constraint, val) works
#=
constraint = Bijection(bijector(Gamma(2,2)))
val = 2.0
check_constraint(constraint, val)
val_u = unconstrain(constraint, val)
val_o = constrain(constraint, val_u)
=#

############################################################################################
# 4. If Objective is used, include a method that computes logabsdet from transformation to unconstrained domain. Same syntax as in Bijectors.jl package is used, i.e., -log_abs_det_jac is returned for computations.
function log_abs_det_jac(b::Bijection, θ::T) where {T}
    #!NOTE: See Bijectors.logabsdetjac implementation for '-'
    #!NOTE Formulation checked in tests/test-bijector
    return -Bijectors.logabsdetjac(b.unconstrain, θ)
end
function log_abs_det_jac(b::Bijection{A,B}, θ::T) where {A<:typeof(identity), B<:typeof(identity), T}
    #!NOTE: Allow Fixed Params of arbitrary size work nice with bijectors
    return 0.0
end

############################################################################################
# 5. Add _check function to check for all other peculiar things that should be tested if new releases come out and add to Test section.
function _check(
    _rng::Random.AbstractRNG,
    b::Bijection,
    val::Union{R,Array{R},AbstractArray},
) where {R<:Real}
    return typeof( unconstrain(b, val) ) == typeof(val) ? true : false
end

############################################################################################
#Export
export
    Bijection,
    constrain,
    unconstrain,
    log_abs_det_jac
