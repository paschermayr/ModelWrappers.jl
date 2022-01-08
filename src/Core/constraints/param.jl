############################################################################################
"""
$(TYPEDEF)
Abstract super type for parameter constraints.
"""
abstract type AbstractConstrained end

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
        ArgCheck.@argcheck _checkparam(_rng, val, constraint) println(
            "Val and constraint do not match for ", val, " and ", constraint, "."
        )
        return new{A,B}(val, constraint)
    end
end
Param(val::A, constraint::B) where {A,B} = Param(Random.GLOBAL_RNG, val, constraint)
flatten(param::Param) = flatten(param.val, param.constraint)


############################################################################################
# Export
export Param
