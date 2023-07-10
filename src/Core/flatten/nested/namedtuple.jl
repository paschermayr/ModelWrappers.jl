############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::NamedTuple{names}
) where {T<:AbstractFloat,F<:FlattenTypes, names}
    _flatten, _unflatten = construct_flatten(T, flattentype, unflattentype, values(x))
    function flatten_to_NamedTuple(x::NamedTuple{names}) where {names}
        return _flatten(values(x))
    end
    function unflatten_to_NamedTuple(v::Union{<:Real,AbstractVector{<:Real}})
        v_vec = _unflatten(v)
        return typeof(x)(v_vec)
    end
    return flatten_to_NamedTuple, unflatten_to_NamedTuple
end

function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    x::NamedTuple{names}
) where {T<:AbstractFloat,F<:FlattenTypes, names}
    _flatten, _unflatten = construct_flatten(T, flattentype, unflattentype, values(x))
    function flatten_to_NamedTuple(x::NamedTuple{names}) where {names}
        return _flatten(values(x))
    end
    function unflatten_to_NamedTupleAD(v::Union{<:Real,AbstractVector{<:Real}})
        v_vec = _unflatten(v)
        #!NOTE: cannot use typeof(x) in AD as Floats are converted to Duals
        return NamedTuple{names}(v_vec)
    end
    return flatten_to_NamedTuple, unflatten_to_NamedTupleAD
end

############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    constraint::NamedTuple,
    x::NamedTuple{names}
) where {T<:AbstractFloat,F<:FlattenTypes, names}
    _flatten, _unflatten = construct_flatten(T, flattentype, unflattentype, values(constraint), values(x))
    function flatten_to_NamedTuple(x::NamedTuple{names}) where {names}
        return _flatten(values(x))
    end
    function unflatten_to_NamedTuple(v::Union{<:Real,AbstractVector{<:Real}})
        v_vec = _unflatten(v)
        return typeof(x)(v_vec)
    end
    return flatten_to_NamedTuple, unflatten_to_NamedTuple
end

function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    constraint::NamedTuple,
    x::NamedTuple{names}
) where {T<:AbstractFloat,F<:FlattenTypes, names}
    _flatten, _unflatten = construct_flatten(T, flattentype, unflattentype, values(constraint), values(x))
    function flatten_to_NamedTuple(x::NamedTuple{names}) where {names}
        return _flatten(values(x))
    end
    function unflatten_to_NamedTupleAD(v::Union{<:Real,AbstractVector{<:Real}})
        v_vec = _unflatten(v)
        #!NOTE: cannot use typeof(x) in AD as Floats are converted to Duals
        return NamedTuple{names}(v_vec)
    end
    return flatten_to_NamedTuple, unflatten_to_NamedTupleAD
end

############################################################################################
#!NOTE: If we map over collections, need to extend this to params.jl functions:
# _check, constrain, unconstrain, log_abs_det_jac

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    constraint::NamedTuple{names1},
    x::NamedTuple{names2}
) where {names1, names2}
    return _check(_rng, values(constraint), values(x))
end

############################################################################################
function constrain(
    constraint::NamedTuple{names1},
    θᵤ::NamedTuple{names2}
) where {names1, names2}
    #!NOTE: ReverseDiff only works with NamedTuple{names} instead of typeof(b⁻¹ᵥ). Takes much more time - might remove ReverseDiff once other ReverseModeAD package comes up.
    # see: https://github.com/JuliaDiff/ReverseDiff.jl/issues/178
#    return typeof(b⁻¹ᵥ)(constrain(values(b⁻¹ᵥ), values(θᵤ)))
    return NamedTuple{names1}(constrain(values(constraint), values(θᵤ)))
end

############################################################################################
function unconstrain(
    constraint::NamedTuple{names1},
    θ::NamedTuple{names2}
) where {names1, names2}
    #!NOTE: ReverseDiff only works with NamedTuple{names} instead of typeof(b⁻¹ᵥ). Takes much more time - might remove ReverseDiff once other ReverseModeAD package comes up.
    # see: https://github.com/JuliaDiff/ReverseDiff.jl/issues/178
#    return typeof(bᵥ)(unconstrain(values(bᵥ), values(θ)))
    return NamedTuple{names1}(unconstrain(values(constraint), values(θ)))
end

############################################################################################
function log_abs_det_jac(
    constraint::NamedTuple{names1},
    θ::NamedTuple{names2}
) where {names1, names2}
    return log_abs_det_jac(values(constraint), values(θ))
end

############################################################################################
#Export
export
    construct_flatten,
    constrain,
    unconstrain,
    log_abs_det_jac
