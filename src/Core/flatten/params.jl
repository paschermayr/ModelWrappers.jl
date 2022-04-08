############################################################################################
"""
$(TYPEDEF)
Abstract super type for parameter constraints.
"""
abstract type AbstractConstraint end

#!NOTE: Needs to be overloaded for new AbstractConstraint
"Check if (constraint, val) combination is valid."
function _check end

"Construct a transform and inverse transform function for given (constraint, val) "
function construct_transform end

############################################################################################
"""
$(TYPEDEF)
Abstract super type for parameter transformations.
"""
abstract type AbstractTransformer end

"Constrain `val` with given `constraint`"
function constrain end

"Unconstrain `val` with given `constraint`"
function unconstrain end

"Compute log(abs(determinant(jacobian(`x`)))) for given transformer."
function log_abs_det_jac end

############################################################################################
"""
$(TYPEDEF)

Utility struct to define Parameter in a way ModelWrappers.jl can handle. Will be separated in ModelWrapper struct for type stability.

# Fields
$(TYPEDFIELDS)
"""
struct Param{A,B}
    val::A
    constraint::B
    function Param(_rng::Random.AbstractRNG, val::A, constraint::B) where {A,B}
        ArgCheck.@argcheck _check(_rng, constraint, val) string(
            "Val and constraint do not match for ", val, " and ", constraint, "."
        )
        return new{A,B}(val, constraint)
    end
end
Param(val::A, constraint::B) where {A,B} = Param(Random.GLOBAL_RNG, val, constraint)



############################################################################################
#!NOTE: Needs to be overloaded for new AbstractTransformer


############################################################################################
# Export
export
    Param,
    AbstractConstraint,
    AbstractTransformer,
    construct_transform,
    constrain,
    unconstrain,
    log_abs_det_jac
