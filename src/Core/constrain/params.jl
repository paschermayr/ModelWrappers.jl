############################################################################################
"""
    $(FUNCTIONNAME)(x )
Check if constrain and unconstrain functions of 'constraint' can map 'val' correctly. Will be called when initiating a 'Param' struct.

# Examples
```julia
```
"""
function check_constraint(constraint::AbstractConstraint, val::V) where {V}
    #Transform val to unconstrained domain
    valᵤ = unconstrain(constraint, val)
    #Constrain to original domain and compare vals
    valₒ = constrain(constraint, valᵤ)
    _check = val ≈ valₒ
    return _check
end

############################################################################################
"""
$(TYPEDEF)

Utility struct to define Parameter in a way ModelWrappers.jl can handle. Will be separated in ModelWrapper struct for type stability.

# Fields
$(TYPEDFIELDS)
"""
struct Param{A,B}
    constraint::A
    val::B
    function Param(_rng::Random.AbstractRNG, constraint::A, val::B) where {A<:AbstractConstraint,B}
        #1 User defined check for all other peculiar things
        ArgCheck.@argcheck _check(_rng, constraint, val) string(
            "Val and constraint do not match for ", val, " and ", constraint, "."
        )
        #2 Generic test if constraint can map val accordingly
        ArgCheck.@argcheck check_constraint(constraint, val) string(
            "'constraint' cannot constrain and unconstrain 'val' to same value. Adjust or change constraint, and/or report an issue on GitHub."
        )
        return new{A,B}(constraint, val)
    end
end
Param(constraint::A, val::B) where {A, B} = Param(Random.GLOBAL_RNG, constraint, val)

############################################################################################
# Export
export
    Param,
    check_constraint
