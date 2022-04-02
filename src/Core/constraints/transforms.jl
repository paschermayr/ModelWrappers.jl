############################################################################################
# A bunch of functions to reduce dimensionality of parameters. Need ChainRulesCore support if mutating.
# !NOTE: flatten/unflatten is purely to change shape of parameter, constrain/unconstrain functions operate separately.
# !NOTE 2: Currently Bijectors.jl handles all transformations, but I am open to integrate dimensionality reductions below for the flatten/unflatten part.
# !NOTE 3: This will work fine even with Bijectors, as long as original dimensionality is mapped back via unflatten.

############################################################################################
"""
$(SIGNATURES)
Flatten array x. Not exported.

# Examples
```julia
```

"""
@inline function flatten_array(mat::AbstractArray{R}) where {R<:Real}
    return vec(mat)
end

"""
$(SIGNATURES)
Fill array with elements of vec. Not exported.

# Examples
```julia
```

"""
@inline function fill_array!(
    buffer::AbstractArray{T}, vec::Union{F,AbstractArray{F}}
) where {T<:Real,F<:Real}
    @inbounds for iter in eachindex(vec)
        buffer[iter] = vec[iter]
    end
    return buffer
end
function ChainRulesCore.rrule(
    ::typeof(fill_array!), mat::AbstractMatrix{R}, v::Union{T,AbstractVector{T}}
) where {R<:Real,T<:Real}
    # forward pass: Fill Matrix with Vector elements
    L = fill_array!(mat, v)
    # backward pass: Fill Vector with Matrix elements
    function pullback_Idx(Δ)
        return ChainRulesCore.NoTangent(),
        ChainRulesCore.unthunk(Δ),
        flatten_array(ChainRulesCore.unthunk(Δ))
    end
    return L, pullback_Idx
end

############################################################################################
"""
$(SIGNATURES)
Flatten vector x to k-1 dimensions. Not exported.

# Examples
```julia
```

"""
function flatten_Simplex(x::AbstractVector{R}) where {R<:Real}
    return x[1:(end - 1)]
end

"""
$(SIGNATURES)
Expand vector of k-1 dimensions back to k dimensions. Not exported.

# Examples
```julia
```

"""
function Simplex_from_flatten(x_vec::Union{R,AbstractVector{R}}) where {R<:Real}
    buffer = zeros(eltype(x_vec), length(x_vec) + 1)
    @inbounds for iter in eachindex(x_vec)
        buffer[iter] = x_vec[iter]
    end
    buffer[end] = 1.0 - sum(x_vec)
    return buffer
end
function ChainRulesCore.rrule(
    ::typeof(Simplex_from_flatten), v::Union{R,AbstractVector{R}}
) where {R<:Real}
    # forward pass: From k-1 to k dimensions
    L = Simplex_from_flatten(v)
    # backward pass: From k to k-1 dimensions
    function pullback_Simplex(Δ)
        return ChainRulesCore.NoTangent(), flatten_Simplex(ChainRulesCore.unthunk(Δ))
    end
    return L, pullback_Simplex
end

"""
$(SIGNATURES)
Inplace version of Simplex_from_flatten. Not exported.

# Examples
```julia
```

"""
function Simplex_from_flatten!(
    buffer::AbstractVector{R}, x_vec::Union{T,AbstractVector{T}}
) where {R<:Real,T<:Real}
    @inbounds for iter in eachindex(x_vec)
        buffer[iter] = x_vec[iter]
    end
    buffer[end] = 1.0 - sum(x_vec)
    return buffer
end
function ChainRulesCore.rrule(
    ::typeof(Simplex_from_flatten!), p::AbstractVector{R}, v::Union{T,AbstractVector{T}}
) where {R<:Real,T<:Real}
    # forward pass: From k-1 to k dimensions
    L = Simplex_from_flatten!(p, v)
    # backward pass: From k to k-1 dimensions
    function pullback_Simplex(Δ)
        return ChainRulesCore.NoTangent(),
        ChainRulesCore.unthunk(Δ),
        flatten_Simplex(ChainRulesCore.unthunk(Δ))
    end
    return L, pullback_Simplex
end

############################################################################################
"""
$(SIGNATURES)
Flatten matrix to vector. Not exported.

# Examples
```julia
```

"""
@inline function flatten_Symmetric(mat::AbstractMatrix{R}, idx::BitMatrix) where {R<:Real}
    return vec(mat[idx])
end

"""
$(SIGNATURES)
Expand vector back to (symmetric) Matrix. Not exported.

# Examples
```julia
```

"""
@inline function Symmetric_from_flatten(
    x_vec::Union{R,AbstractVector{R}}, idx::BitMatrix
) where {R<:Real}
    ArgCheck.@argcheck size(idx, 1) == size(idx, 2) "'idx' has to be a square matrix."
    mat = zeros(eltype(x_vec), size(idx))
    counter = 0
    @inbounds @simd for Ncol in axes(mat, 2)
        for Nrow in axes(mat, 1)
            if idx[Nrow, Ncol]
                counter += 1
                mat[Nrow, Ncol] = mat[Ncol, Nrow] = x_vec[counter]
            end
        end
    end
    return mat
end
function ChainRulesCore.rrule(
    ::typeof(Symmetric_from_flatten), v::Union{R,AbstractVector{R}}, idx::BitMatrix
) where {R<:Real}
    # forward pass: Fill Matrix with Vector elements
    L = Symmetric_from_flatten(v, idx)
    # backward pass: Fill Vector with Matrix elements
    function pullback_Idx(Δ)
        return ChainRulesCore.NoTangent(),
        flatten_Symmetric(ChainRulesCore.unthunk(Δ), idx),
        ChainRulesCore.NoTangent()
    end
    return L, pullback_Idx
end

"""
$(SIGNATURES)
Inplace version of Symmetric_from_flatten. Not exported.

# Examples
```julia
```

"""
@inline function Symmetric_from_flatten!(
    mat::AbstractMatrix{T}, x_vec::Union{R,AbstractVector{R}}, idx::BitMatrix
) where {T<:Real,R<:Real}
    ArgCheck.@argcheck size(mat) == size(idx) "Dimension of matrix and entries do not match."
    counter = 0
    @inbounds @simd for Ncol in axes(mat, 2)
        for Nrow in axes(mat, 1)
            if idx[Nrow, Ncol]
                counter += 1
                mat[Nrow, Ncol] = mat[Ncol, Nrow] = x_vec[counter]
            end
        end
    end
    return mat
end
function ChainRulesCore.rrule(
    ::typeof(Symmetric_from_flatten!),
    mat::AbstractMatrix{T},
    v::Union{R,AbstractVector{R}},
    idx::BitMatrix,
) where {T<:Real,R<:Real}
    # forward pass: Fill Matrix with Vector elements
    L = Symmetric_from_flatten!(mat, v, idx)
    # backward pass: Fill Vector with Matrix elements
    function pullback_Idx(Δ)
        return ChainRulesCore.NoTangent(),
        ChainRulesCore.unthunk(Δ), flatten_Symmetric(ChainRulesCore.unthunk(Δ), idx),
        ChainRulesCore.NoTangent()
    end
    return L, pullback_Idx
end

############################################################################################
"""
$(SIGNATURES)
Assign subset of elements to track in Matrix mat. Not exported.

# Examples
```julia
```

"""
@inline function tag(
    mat::AbstractMatrix{R}, upper::Bool=true, diag::Bool=true
) where {R<:Real}
    ArgCheck.@argcheck size(mat, 1) == size(mat, 2) "Dimension of matrix and entries do not match."
    idx =
        upper ? LinearAlgebra.triu(trues(size(mat))) : LinearAlgebra.tril(trues(size(mat)))
    if !diag
        @inbounds for iter in axes(idx, 1)
            idx[iter, iter] = 0.0
        end
    end
    return idx
end

############################################################################################
# Export
