############################################################################################
"""
$(SIGNATURES)
Check if 'θ' is of finite value and return Bool. Not exported.

# Examples
```julia
```

"""
function _checkfinite(θ::T, max_val::R=max_val) where {T<:Real,R<:Real}
    return isfinite(θ) && θ < max_val
end
function _checkfinite(θ::Array{T}, max_val::R=max_val) where {T<:Real,R<:Real}
    @inbounds @simd for iter in eachindex(θ)
        if !_checkfinite(θ[iter], max_val)
            return false
        end
    end
    return true
end
function _checkfinite(θ::AbstractArray, max_val::R=max_val) where {R<:Real}
    @inbounds @simd for iter in eachindex(θ)
        if !_checkfinite(θ[iter], max_val)
            return false
        end
    end
    return true
end
function _checkfinite(θ::Tuple, max_val::R=max_val) where {R<:Real}
    return all(map(_checkfinite, θ))
end
function _checkfinite(θ::N, max_val::R=max_val) where {N<:NamedTuple} where {R<:Real}
    return all(map(_checkfinite, θ))
end

############################################################################################
"""
$(SIGNATURES)
Check if 'prior' is a valid density and return Bool. Not exported.

# Examples
```julia
```

"""
function _checkprior(prior)
    return false
end
function _checkprior(prior::S) where {S<:Distributions.Distribution}
    return true
end
function _checkprior(prior::Array{T}) where {T}
    @inbounds @simd for iter in eachindex(prior)
        if !_checkprior(prior[iter])
            return false
        end
    end
    return true
end
function _checkprior(prior::AbstractArray)
    @inbounds @simd for iter in eachindex(prior)
        if !_checkprior(prior[iter])
            return false
        end
    end
    return true
end
function _checkprior(prior::N) where {N<:NamedTuple}
    return all(map(_checkprior, prior))
end

############################################################################################
#=
"""
$(SIGNATURES)
Check if all keys of 'x' and 'y' match - works with Nested Tuples - and return Bool. Not exported.

# Examples
```julia
```

"""
function _checkkeys(x::NamedTuple{Kx,Tx}, y::NamedTuple{Ky,Ty}) where {Kx,Tx,Ky,Ty}
    ## Check if all keys in x and y are the same
    if !all(haskey(x, sym) == haskey(y, sym) for sym in keys(x)) ||
       !all(haskey(x, sym) == haskey(y, sym) for sym in keys(y))
        return false
    end
    ## Check if any field in x/y is a NamedTuple itself -> if true, check that field first
    @inbounds @simd for sym in keys(x)
        if isa(x[sym], NamedTuple)
            if !_checkkeys(x[sym], y[sym])
                return false
            end
        end
    end
    ## Else return true
    return true
end
=#
############################################################################################
"""
$(SIGNATURES)
Check if argument is not fixed. Returns NamedTuple with true/false. Needed in addition to _checkprior for nested NamedTuples. Not exported.

# Examples
```julia
```

"""
function _checksampleable(constraint::Fixed)
    return false
end
function _checksampleable(constraint::S) where {S<:Distributions.Distribution}
    return true
end
function _checksampleable(constraint::Vector{T}) where {T}
    @inbounds @simd for iter in eachindex(constraint)
        if !_checksampleable(constraint[iter])
            return false
        end
    end
    return true
end
function _checksampleable(constraint::AbstractArray)
    @inbounds @simd for iter in eachindex(constraint)
        if !_checksampleable(constraint[iter])
            return false
        end
    end
    return true
end
function _checksampleable(constraint::NamedTuple{names}) where {names}
    return NamedTuple{names}(map(_checksampleable, constraint))
end

############################################################################################
"""
$(SIGNATURES)
Check if provided val-constraint combination is valid for Param struct. Not exported.

# Examples
```julia
```

"""
function _checkparam(
    _rng::Random.AbstractRNG, val::AbstractArray, constraint::AbstractArray
)
    return all(
        map(val, constraint) do valᵢ, constraintᵢ
            _checkparam(_rng, valᵢ, constraintᵢ)
        end,
    )
end
_checkparam(val, constraint) = _checkparam(Random.GLOBAL_RNG, val, constraint)

############################################################################################
"""
$(SIGNATURES)
Check if all values in (Nested) NamedTuple are a 'Param' struct and return Bool. Not exported.

# Examples
```julia
```

"""
function _checkparams(param)
    return false
end
function _checkparams(param::Param)
    return true
end

function _checkparams(param::N) where {N<:NamedTuple}
    return all(map(_checkparams, param))
end

############################################################################################
# Export
