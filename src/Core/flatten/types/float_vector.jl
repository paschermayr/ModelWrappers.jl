############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::AbstractVector{R}
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    len = length(x)
    buffer_flat = zeros(T, size(x))
    function flatten_to_Real_Vec(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len
        return fill_array!(buffer_flat, v)
    end
    buffer_unflat = zeros(R, size(x))
    function unflatten_to_Real_Vec(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len
        return fill_array!(buffer_unflat, v)
    end
    return flatten_to_Real_Vec, unflatten_to_Real_Vec
end

function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    x::AbstractVector{R}
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    len = length(x)
    function flatten_to_Real_Vec_AD(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len
        return identity(v)
    end
    function unflatten_to_Real_Vec_AD(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len
        return identity(v)
    end
    return flatten_to_Real_Vec_AD, unflatten_to_Real_Vec_AD
end

############################################################################################
#Export
export
    construct_flatten
