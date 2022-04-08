############################################################################################
"""
$(TYPEDEF)
Contains information about parameter distributions, transformations and constraints.

# Fields
$(TYPEDFIELDS)
"""
struct ParameterInfo{C,R<:ReConstructor,T<:TransformConstructor}
    "Constraint distribution/boundaries for all model parameter."
    constraint::C
    "Contains information for flatten/unflatten parameter"
    reconstruct::R
    "Contains information for constraining and unconstraining parameter."
    transform::T
    function ParameterInfo(
        constraint::C, val::B, flattendefault::D
    ) where {C<:NamedTuple,B<:NamedTuple,D<:FlattenDefault}
        ## Create flatten constructor
        constructor = ReConstructor(flattendefault, constraint, val)
        ## Assign transformer constraint NamedTuple
        transformer = TransformConstructor(constraint, val)
        ## Return ParameterInfo
        return new{C,typeof(constructor),typeof(transformer)}(
            constraint, constructor, transformer
        )
    end
end

############################################################################################
#export
export ParameterInfo
