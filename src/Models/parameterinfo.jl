############################################################################################
"""
$(TYPEDEF)
Contains information about parameter distributions, transformations and constraints.

# Fields
$(TYPEDFIELDS)
"""
struct ParameterInfo{A,B,C,D<:FlattenDefault,E<:Function,F<:Function}
    "Constraint distribution/boundaries for all model parameter."
    constraint::A
    "Bijector for all model parameter."
    b::B
    "Inverse-Bijector for all model parameter."
    b⁻¹::C
    "Default Flattening setting"
    flattendefault::D
    "Function to unflatten model parameter, if provided as a vector."
    unflatten::E
    "Function to unflatten model parameter, if provided as a vector."
    unflatten_AD::F
    function ParameterInfo(
        val::B, constraint::A, flattendefault::D
    ) where {A<:NamedTuple,B<:NamedTuple,D<:FlattenDefault}
        @unpack output, flattentype = flattendefault
        ## Create unflatten function
        flatten_strict = FlattenDefault(output, flattentype, UnflattenStrict())
        flatten_AD = FlattenDefault(output, flattentype, UnflattenAD())
        _, unflatten_func = flatten(flatten_strict, val, constraint)
        _, unflatten_func_AD = flatten(flatten_AD, val, constraint)
        ## Assign bijector and inv_bijector to constraint NamedTuple
        b = _to_bijector(constraint)
        b⁻¹ = _to_inv_bijector(b)
        ## Return ParameterInfo
        return new{
            A,typeof(b),typeof(b⁻¹),D,typeof(unflatten_func),typeof(unflatten_func_AD)
        }(
            constraint, b, b⁻¹, flattendefault, unflatten_func, unflatten_func_AD
        )
    end
end
############################################################################################
#export
export ParameterInfo
