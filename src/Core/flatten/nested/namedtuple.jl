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
# _check, construct_transform, constrain, unconstrain, log_abs_det_jac

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    constraint::NamedTuple{names1},
    x::NamedTuple{names2}
) where {names1, names2}
    return _check(_rng, values(constraint), values(x))
end

############################################################################################
function construct_transform(
    constraint::NamedTuple{names1},
    x::NamedTuple{names2}
) where {names1, names2}
## Obtain transform/inversetransform constructor for each element
    _transform, _inversetransform = construct_transform(values(constraint), values(x))
## Return flatten/unflatten for AbstractArray
    return NamedTuple{names1}(_transform), NamedTuple{names1}(_inversetransform)
end

############################################################################################
function constrain(
    b????????::NamedTuple{names1},
    ?????::NamedTuple{names2}
) where {names1, names2}
    #!NOTE: ReverseDiff only works with NamedTuple{names} instead of typeof(b????????). Takes much more time - might remove ReverseDiff once other ReverseModeAD package comes up.
    # see: https://github.com/JuliaDiff/ReverseDiff.jl/issues/178
    return NamedTuple{names1}(constrain(values(b????????), values(?????)))
end

############################################################################################
function unconstrain(
    b???::NamedTuple{names1},
    ??::NamedTuple{names2}
) where {names1, names2}
    #!NOTE: ReverseDiff only works with NamedTuple{names} instead of typeof(b????????). Takes much more time - might remove ReverseDiff once other ReverseModeAD package comes up.
    # see: https://github.com/JuliaDiff/ReverseDiff.jl/issues/178
    return NamedTuple{names1}(unconstrain(values(b???), values(??)))
end

############################################################################################
function log_abs_det_jac(
    b???::NamedTuple{names1},
    ??::NamedTuple{names2}
) where {names1, names2}
    return log_abs_det_jac(values(b???), values(??))
end

############################################################################################
#Export
export
    construct_flatten,
    construct_transform,
    constrain,
    unconstrain,
    log_abs_det_jac
