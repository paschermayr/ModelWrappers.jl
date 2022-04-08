############################################################################################
"""
$(SIGNATURES)
Recursively collect constraints of 'Param' struct. Not exported.

# Examples
```julia
```

"""
function _get_constraint(param::Param)
    return param.constraint
end
function _get_constraint(val::NamedTuple{names}) where {names}
    return NamedTuple{names}(Tuple(map(_get_constraint, val)))
end
"""
$(SIGNATURES)
Recursively collect values of 'Param' struct. Not exported.

# Examples
```julia
```

"""
function _get_val(param::Param)
    return param.val
end
function _get_val(val::NamedTuple{names}) where {names}
    return NamedTuple{names}(Tuple(map(_get_val, val)))
end

############################################################################################
"""
$(SIGNATURES)
Returns NamedTuple of true/false given parameter is not fixed. Not exported.

# Examples
```julia
```

"""
function _allparam(val::Bool)
    return val
end
function _allparam(val::A) where {A<:NamedTuple}
    return all(map(_allparam, val))
end
"""
$(SIGNATURES)
Returns NamedTuple of true/false given parameter is not fixed. Not exported.

# Examples
```julia
```

"""
_anyparam(val::Bool) = _allparam(val)
function _anyparam(val::A) where {A<:NamedTuple}
    return any(map(_anyparam, val))
end

############################################################################################
"""
$(SIGNATURES)
Return parameter names as a string in increasing order. Not exported.

# Examples
```julia
```

"""
function _paramnames(sym::Symbol, len::Integer)
    return [len == 1 ? string(sym) : string(sym, i) for i in Base.OneTo(len)]
end
function _paramnames(sym::NTuple{N,Symbol}, len::NTuple{N,Integer}) where {N}
    ArgCheck.@argcheck length(sym) == length(len)
    return [
        len[n] == 1 ? string(sym[n]) : string(sym[n], i)
        for n in Base.OneTo(length(sym)) for i in Base.OneTo(len[n])
    ]
end
"""
$(SIGNATURES)
Return all parameter names in increasing order. Not exported.

# Examples
```julia
```

"""
function paramnames(
    sym::Symbol,
    types::F,
    constraint,
    val
) where {F<:FlattenDefault}
    flatten, _ = construct_flatten(types, constraint, val)
    val_flattened = flatten(val)
    return _paramnames(sym, length(val_flattened))
end
function paramnames(
    sym::NTuple{N,Symbol},
    types::F,
    constraint,
    val,
) where {F<:FlattenDefault,N}
    flatten, unflatten = construct_flatten(types, constraint, val)
    return _paramnames(sym, unflatten._unflatten.lngth)
end
function paramnames(
    types::F,
    constraint::NamedTuple,
    val::NamedTuple{names},
) where {F<:FlattenDefault,N,names}
    return reduce(
        vcat, map(fld -> paramnames(fld, types, constraint[fld], val[fld]), names)
    )
end

############################################################################################
"""
$(SIGNATURES)
Count length of nested parameter tuple. Not exported.

# Examples
```julia
```

"""
function paramcount(sym::Symbol, types::F, val) where {F<:FlattenDefault}
    _flatten, _ = construct_flatten(types, val)
    val_flattened = _flatten(val)
    return length(val_flattened)
end
function paramcount(sym::NTuple{N,Symbol}, types::F, val) where {F<:FlattenDefault,N}
    _flatten, _unflatten = construct_flatten(types, val)
    return _unflatten._unflatten.lngth
end
function paramcount(types::F, val::NamedTuple{names}) where {F<:FlattenDefault,names}
    return map(fld -> paramcount(fld, types, val[fld]), names)
end

############################################################################################
# Export
