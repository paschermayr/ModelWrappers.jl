############################################################################################
#=
!NOTE: At the moment this only holds all constraints of parameter.
But structuring this way is easier to add features later on as additional fields without changing code base in other packages.
=#
"""
    $(FUNCTIONNAME)(x )
Contains information to constrain and unconstrain parameter for all parameter.
# Examples
```julia
```
"""
struct TransformConstructor{S}
    constraint::S
    function TransformConstructor(constraint::S, val::V) where {S, V}
        return new{S}(constraint)
    end
end

function constrain(transform::T, valᵤ::V) where {T<:TransformConstructor, V}
    val = constrain(transform.constraint, valᵤ)
    return val
end
function unconstrain(transform::T, val::V) where {T<:TransformConstructor, V}
    valᵤ = unconstrain(transform.constraint, val)
    return valᵤ
end
function log_abs_det_jac(transform::T, val::V) where {T<:TransformConstructor, V}
    return log_abs_det_jac(transform.constraint, val)
end

############################################################################################
#export
export TransformConstructor,
    constrain,
    unconstrain,
    log_abs_det_jac