var documenterSearchIndex = {"docs":
[{"location":"intro/#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"intro/","page":"Introduction","title":"Introduction","text":"Yet to be properly done.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ModelWrappers","category":"page"},{"location":"#ModelWrappers","page":"Home","title":"ModelWrappers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ModelWrappers.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ModelWrappers]","category":"page"},{"location":"#ModelWrappers.max_val","page":"Home","title":"ModelWrappers.max_val","text":"Maximum value after transformation before tagged as non-finite.\n\n\n\n\n\n","category":"constant"},{"location":"#ModelWrappers.min_Δ","page":"Home","title":"ModelWrappers.min_Δ","text":"Smallest decrease allowed in the log objective results before tagged as divergent.\n\n\n\n\n\n","category":"constant"},{"location":"#ModelWrappers.AbstractConstraint","page":"Home","title":"ModelWrappers.AbstractConstraint","text":"abstract type AbstractConstraint\n\nAbstract super type for parameter constraints.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.AbstractInitialization","page":"Home","title":"ModelWrappers.AbstractInitialization","text":"abstract type AbstractInitialization\n\nAbstract method to initialize parameter for individual kernels.\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.BaseModel","page":"Home","title":"ModelWrappers.BaseModel","text":"struct BaseModel <: ModelName\n\nDefault modelname of Baytes.Model struct.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Constrained","page":"Home","title":"ModelWrappers.Constrained","text":"struct Constrained{B<:Bijection} <: AbstractConstraint\n\nUtility struct to help assign boundaries to parameter - keeps scalar parameter constrained.\n\nFields\n\nbijection::Bijection\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Fixed","page":"Home","title":"ModelWrappers.Fixed","text":"struct Fixed{B<:Bijection} <: AbstractConstraint\n\nUtility struct to help assign boundaries to parameter - keeps parameter fixed. Useful for assigning buffer values for functions of parameter.\n\nFields\n\nbijection::Bijection\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.FlattenConstructor","page":"Home","title":"ModelWrappers.FlattenConstructor","text":"struct FlattenConstructor{S<:Function, T<:Function}\n\nContains information for flatten construct.\n\nFields\n\nstrict::Function\nflexible::Function\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.FlattenDefault","page":"Home","title":"ModelWrappers.FlattenDefault","text":"struct FlattenDefault{T<:AbstractFloat, F<:FlattenTypes}\n\nDefault arguments for flatten function.\n\nFields\n\noutput::Type{T} where T<:AbstractFloat\nType of flatten output\nflattentype::FlattenTypes\nDetermines if all inputs are flattened (FlattenAll) or only continuous values (FlattenContinuous).\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.FlattenTypes","page":"Home","title":"ModelWrappers.FlattenTypes","text":"abstract type FlattenTypes\n\nSupertype for dispatching different types of flatten. Determines if all inputs are flattened (FlattenAll) or only continuous values (FlattenContinuous).\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.ModelName","page":"Home","title":"ModelWrappers.ModelName","text":"abstract type ModelName\n\nAbstract super type for Baytes Models.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.ModelWrapper","page":"Home","title":"ModelWrappers.ModelWrapper","text":"mutable struct ModelWrapper{M<:(Union{ModelName, P} where P), A<:NamedTuple, C<:NamedTuple, B<:ParameterInfo} <: BaytesCore.AbstractModelWrapper\n\nBaytes Model struct.\n\nContains information about current Model value, name, and information, see also ParameterInfo.\n\nFields\n\nval::NamedTuple\nCurrent Model values as NamedTuple - works with Nested Tuples.\narg::NamedTuple\nSupplementary arguments for log target function that are fixed and dont need to be stored in a trace.\ninfo::ParameterInfo\nInformation about parameter distributions, transformations and constraints, see ParameterInfo.\nid::Union{ModelName, P} where P\nModel id, per default BaseModel. Useful for dispatching ModelWrapper struct.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.NoInitialization","page":"Home","title":"ModelWrappers.NoInitialization","text":"Use current model.val parameter as initial parameter\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Objective","page":"Home","title":"ModelWrappers.Objective","text":"struct Objective{M<:ModelWrapper, D, T<:Tagged, F<:AbstractFloat} <: BaytesCore.AbstractObjective\n\nFunctor to calculate 'ℓfunc' and gradient at unconstrained 'θᵤ', including eventual Jacobian adjustments.\n\nFields\n\nmodel::ModelWrapper\ndata::Any\ntagged::Tagged\ntemperature::AbstractFloat\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Objective-Tuple{NamedTuple, Any, Any}","page":"Home","title":"ModelWrappers.Objective","text":"Functor to call target function for Model given parameter and data. Default method to be used in Automatic Differentiation. model.arg and data are arguments so they can be declared as constant with Enzyme AD engine.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.Objective-Tuple{NamedTuple}","page":"Home","title":"ModelWrappers.Objective","text":"Functor to call target function for Model given parameter and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.OptimInitialization","page":"Home","title":"ModelWrappers.OptimInitialization","text":"Use custom optimization technique for initialization.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Param","page":"Home","title":"ModelWrappers.Param","text":"struct Param{A, B}\n\nUtility struct to define Parameter in a way ModelWrappers.jl can handle. Will be separated in ModelWrapper struct for type stability.\n\nFields\n\nconstraint::Any\nval::Any\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.ParameterInfo","page":"Home","title":"ModelWrappers.ParameterInfo","text":"struct ParameterInfo{R<:ReConstructor, U<:ReConstructor, T<:TransformConstructor}\n\nContains information about parameter distributions, transformations and constraints.\n\nFields\n\nreconstruct::ReConstructor\nContains information for flatten/unflatten parameter\nreconstructᵤ::ReConstructor\nContains information to reconstruct unconstrained parameter - important for non-bijective transformations\ntransform::TransformConstructor\nContains information for constraining and unconstraining parameter.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.PriorInitialization","page":"Home","title":"ModelWrappers.PriorInitialization","text":"Sample (up to Ntrials) times from prior and check if log target distribution is finite at proposed parameter in unconstrained space.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.ReConstructor","page":"Home","title":"ModelWrappers.ReConstructor","text":"struct ReConstructor{F<:FlattenDefault, S<:FlattenConstructor, T<:UnflattenConstructor}\n\nContains information for flatten/unflatten construct.\n\nFields\n\ndefault::FlattenDefault\nflatten::FlattenConstructor\nunflatten::UnflattenConstructor\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Tagged","page":"Home","title":"ModelWrappers.Tagged","text":"struct Tagged{A<:NamedTuple, B<:ParameterInfo}\n\nStores information for a subset of 'model' parameter.\n\nFields\n\nparameter::NamedTuple\nSubset of ModelWrapper parameter names.\ninfo::ParameterInfo\nInformation about subset of parameter distributions, transformations and constraints, see ParameterInfo.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.TransformConstructor","page":"Home","title":"ModelWrappers.TransformConstructor","text":"TransformConstructor(x )\n\nContains information to constrain and unconstrain parameter for all parameter.\n\nExamples\n\n\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Unconstrained","page":"Home","title":"ModelWrappers.Unconstrained","text":"struct Unconstrained{B<:Bijection} <: AbstractConstraint\n\nUtility struct to help assign boundaries to parameter - keeps parameter unconstrained. Useful for assigning buffer values for functions of parameter.\n\nFields\n\nbijection::Bijection\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.UnflattenConstructor","page":"Home","title":"ModelWrappers.UnflattenConstructor","text":"struct UnflattenConstructor{S<:Function, T<:Function}\n\nContains information for unflatten construct.\n\nFields\n\nstrict::Function\nflexible::Function\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.UnflattenTypes","page":"Home","title":"ModelWrappers.UnflattenTypes","text":"abstract type UnflattenTypes\n\nDetermines if unflatten returns original type or if type may change (AD friendly).\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#Base.fill!-Tuple{ModelWrapper, NamedTuple}","page":"Home","title":"Base.fill!","text":"fill!(model, θ)\n\n\nInplace version of fill.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Base.fill-Tuple{ModelWrapper, NamedTuple}","page":"Home","title":"Base.fill","text":"fill(model, θ)\n\n\nFill 'model' values with NamedTuple 'θ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Base.print-Union{Tuple{ModelWrapper}, Tuple{S}, Tuple{ModelWrapper, S}} where S<:Union{Symbol, Tuple{Vararg{Symbol, k}} where k}","page":"Home","title":"Base.print","text":"print(model)\nprint(model, params)\n\n\nPrint 'model' parameter values and constraints of symbols 'params'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.generate-Tuple{Random.AbstractRNG, Objective}","page":"Home","title":"BaytesCore.generate","text":"generate(_rng, objective)\n\n\nGenerate statistics given model parameter and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.generate_showvalues-Tuple{ModelWrapper}","page":"Home","title":"BaytesCore.generate_showvalues","text":"generate_showvalues(model)\n\n\nShow current values of Model as NamedTuple\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.Simplex_from_flatten!-Union{Tuple{T}, Tuple{R}, Tuple{AbstractVector{R}, Union{AbstractVector{T}, T}}} where {R<:Real, T<:Real}","page":"Home","title":"ModelWrappers.Simplex_from_flatten!","text":"Simplex_from_flatten!(buffer, x_vec)\n\n\nInplace version of Simplexfromflatten. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.Simplex_from_flatten-Union{Tuple{Union{AbstractVector{R}, R}}, Tuple{R}} where R<:Real","page":"Home","title":"ModelWrappers.Simplex_from_flatten","text":"Expand vector of k-1 dimensions back to k dimensions. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.Symmetric_from_flatten!-Union{Tuple{R}, Tuple{T}, Tuple{AbstractMatrix{T}, Union{AbstractVector{R}, R}, BitMatrix}} where {T<:Real, R<:Real}","page":"Home","title":"ModelWrappers.Symmetric_from_flatten!","text":"Symmetric_from_flatten!(mat, x_vec, idx)\n\n\nInplace version of Symmetricfromflatten. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.Symmetric_from_flatten-Union{Tuple{R}, Tuple{Union{AbstractVector{R}, R}, BitMatrix}} where R<:Real","page":"Home","title":"ModelWrappers.Symmetric_from_flatten","text":"Symmetric_from_flatten(x_vec, idx)\n\n\nExpand vector back to (symmetric) Matrix. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._allparam-Tuple{Bool}","page":"Home","title":"ModelWrappers._allparam","text":"_allparam(val)\n\n\nReturns NamedTuple of true/false given parameter is not fixed. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._anyparam-Tuple{Bool}","page":"Home","title":"ModelWrappers._anyparam","text":"_anyparam(val)\n\n\nReturns NamedTuple of true/false given parameter is not fixed. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._check-Tuple{Any, Any, Any}","page":"Home","title":"ModelWrappers._check","text":"Check if (constraint, val) combination is valid. If nothing else specified, returns false (!) per default so check necessary.\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._checkfinite-Union{Tuple{R}, Tuple{T}, Tuple{T, R}} where {T<:Real, R<:Real}","page":"Home","title":"ModelWrappers._checkfinite","text":"_checkfinite(θ)\n_checkfinite(θ, max_val)\n\n\nCheck if 'θ' is of finite value and return Bool. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._checkparams-Tuple{Any}","page":"Home","title":"ModelWrappers._checkparams","text":"_checkparams(param)\n\n\nCheck if all values in (Nested) NamedTuple are a 'Param' struct and return Bool. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._checkprior-Tuple{Any}","page":"Home","title":"ModelWrappers._checkprior","text":"_checkprior(prior)\n\n\nCheck if 'prior' is a valid density and return Bool. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._checksampleable-Tuple{Any}","page":"Home","title":"ModelWrappers._checksampleable","text":"_checksampleable(constraint)\n\n\nCheck if argument is not fixed. Returns NamedTuple with true/false. Needed in addition to _checkprior for nested NamedTuples. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._get_constraint-Tuple{Param}","page":"Home","title":"ModelWrappers._get_constraint","text":"_get_constraint(param)\n\n\nRecursively collect constraints of 'Param' struct. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._get_val-Tuple{Param}","page":"Home","title":"ModelWrappers._get_val","text":"_get_val(param)\n\n\nRecursively collect values of 'Param' struct. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._paramnames-Tuple{Symbol, Integer}","page":"Home","title":"ModelWrappers._paramnames","text":"_paramnames(sym, len)\n\n\nReturn parameter names as a string in increasing order. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.check_constraint-Union{Tuple{V}, Tuple{AbstractConstraint, V}} where V","page":"Home","title":"ModelWrappers.check_constraint","text":"check_constraint(x )\n\nCheck if constrain and unconstrain functions of 'constraint' can map 'val' correctly. Will be called when initiating a 'Param' struct.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.constrain","page":"Home","title":"ModelWrappers.constrain","text":"Constrain val with given constraint\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.constrain!-Tuple{AbstractConstraint, Any}","page":"Home","title":"ModelWrappers.constrain!","text":"Inplace constrain val with given constraint, using 'val' as buffer.\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.constrain-Tuple{ModelWrapper, NamedTuple}","page":"Home","title":"ModelWrappers.constrain","text":"constrain(model, θ)\n\n\nConstrain 'θᵤ' values with model.info ParameterInfo.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.constrain-Tuple{ModelWrapper, Tagged, NamedTuple}","page":"Home","title":"ModelWrappers.constrain","text":"constrain(model, tagged, θ)\n\n\nConstrain 'θᵤ' values with tagged.info ParameterInfo.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.construct_flatten","page":"Home","title":"ModelWrappers.construct_flatten","text":"construct_flatten(x ) Construct a flatten function for 'x' given specifications in 'df'.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.dynamics-Tuple{Objective}","page":"Home","title":"ModelWrappers.dynamics","text":"dynamics(objective)\n\n\nAssign model dynamics for a given objective.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.fill_array!-Union{Tuple{F}, Tuple{T}, Tuple{AbstractArray{T}, Union{AbstractArray{F}, F}}} where {T<:Real, F<:Real}","page":"Home","title":"ModelWrappers.fill_array!","text":"fill_array!(buffer, vec)\n\n\nFill array with elements of vec. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.flatten","page":"Home","title":"ModelWrappers.flatten","text":"flatten(x )\n\nConvert 'x' into a Vector.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.flatten-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.flatten","text":"flatten(model)\n\n\nFlatten 'model' values and return as vector.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.flattenAD","page":"Home","title":"ModelWrappers.flattenAD","text":"flattenAD(x )\n\nConvert 'x' into a Vector that is AD compatible.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.flatten_Simplex-Union{Tuple{AbstractVector{R}}, Tuple{R}} where R<:Real","page":"Home","title":"ModelWrappers.flatten_Simplex","text":"flatten_Simplex(x)\n\n\nFlatten vector x to k-1 dimensions. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.flatten_Symmetric-Union{Tuple{R}, Tuple{AbstractMatrix{R}, BitMatrix}} where R<:Real","page":"Home","title":"ModelWrappers.flatten_Symmetric","text":"flatten_Symmetric(mat, idx)\n\n\nFlatten matrix to vector. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.flatten_array-Union{Tuple{AbstractArray{R}}, Tuple{R}} where R<:Real","page":"Home","title":"ModelWrappers.flatten_array","text":"flatten_array(mat)\n\n\nFlatten array x. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_abs_det_jac","page":"Home","title":"ModelWrappers.log_abs_det_jac","text":"Compute log(abs(determinant(jacobian(x)))) for given transformer to unconstrained (!) domain.\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.log_abs_det_jac-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.log_abs_det_jac","text":"log_abs_det_jac(model)\n\n\nEvaluate eventual Jacobian adjustments from transformations at 'model' values.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_prior-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.log_prior","text":"log_prior(model)\n\n\nEvaluate Log density of 'model' prior given current 'model' values.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_prior-Union{Tuple{T}, Tuple{Any, T}} where T","page":"Home","title":"ModelWrappers.log_prior","text":"Evaluate Log density of 'prior' at 'θ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_prior_with_transform-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.log_prior_with_transform","text":"log_prior_with_transform(model)\n\n\nEvaluate Log density and eventual Jacobian adjustments of 'model' prior given current 'model' values.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_prior_with_transform-Union{Tuple{T}, Tuple{Any, T}} where T","page":"Home","title":"ModelWrappers.log_prior_with_transform","text":"log_prior_with_transform(prior, θ)\n\n\nEvaluate Log density and eventual Jacobian adjustments from transformation of 'prior' at 'θ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.paramcount-Union{Tuple{F}, Tuple{Symbol, F, Any}} where F<:FlattenDefault","page":"Home","title":"ModelWrappers.paramcount","text":"paramcount(sym, types, val)\n\n\nCount length of nested parameter tuple. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.paramnames-Union{Tuple{F}, Tuple{Symbol, F, Any, Any}} where F<:FlattenDefault","page":"Home","title":"ModelWrappers.paramnames","text":"paramnames(sym, types, constraint, val)\n\n\nReturn all parameter names in increasing order. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.predict-Tuple{Random.AbstractRNG, Objective}","page":"Home","title":"ModelWrappers.predict","text":"predict(_rng, objective)\n\n\nPredict new data given model parameter and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.predictive-Tuple{Random.AbstractRNG, Objective, PriorInitialization, Integer}","page":"Home","title":"ModelWrappers.predictive","text":"predictive(_rng, objective, init, iter)\n\n\nUse Prior predictive samples to check model assumptions. Needs dispatch on simulate(rng, model).\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.sample_constraint-Tuple{Random.AbstractRNG, Any, Any}","page":"Home","title":"ModelWrappers.sample_constraint","text":"sample_constraint(_rng, prior, val)\n\n\nSample from constraint if 'prior'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.simulate","page":"Home","title":"ModelWrappers.simulate","text":"Simulate data given Model parameter.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.tag-Union{Tuple{AbstractMatrix{R}}, Tuple{R}, Tuple{AbstractMatrix{R}, Bool}, Tuple{AbstractMatrix{R}, Bool, Bool}} where R<:Real","page":"Home","title":"ModelWrappers.tag","text":"tag(mat)\ntag(mat, upper)\ntag(mat, upper, diag)\n\n\nAssign subset of elements to track in Matrix mat. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unconstrain","page":"Home","title":"ModelWrappers.unconstrain","text":"Unconstrain val with given constraint\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.unconstrain!-Tuple{AbstractConstraint, Any}","page":"Home","title":"ModelWrappers.unconstrain!","text":"Inplace unconstrain val with given constraint, using 'val' as buffer.\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unconstrain-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.unconstrain","text":"unconstrain(model)\n\n\nUnconstrain 'model' values and return as NamedTuple.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unconstrain_flatten-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.unconstrain_flatten","text":"unconstrain_flatten(model)\n\n\nFlatten and unconstrain 'model' values and return as vector.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unflatten","page":"Home","title":"ModelWrappers.unflatten","text":"unflatten(x )\n\nUnflatten 'x' into original shape.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.unflatten!-Union{Tuple{T}, Tuple{ModelWrapper, AbstractVector{T}}} where T<:Real","page":"Home","title":"ModelWrappers.unflatten!","text":"unflatten!(model, θ)\n\n\nInplace version of unflatten.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unflatten-Union{Tuple{T}, Tuple{ModelWrapper, AbstractVector{T}}} where T<:Real","page":"Home","title":"ModelWrappers.unflatten","text":"Unlatten Vector 'θ' given constraints from 'model' and return as NamedTuple.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unflattenAD","page":"Home","title":"ModelWrappers.unflattenAD","text":"unflattenAD(x )\n\nUnflatten 'x' into original shape but keep type information of 'x' for AD compatibility.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.unflatten_constrain!-Union{Tuple{T}, Tuple{ModelWrapper, AbstractVector{T}}} where T<:Real","page":"Home","title":"ModelWrappers.unflatten_constrain!","text":"Inplace version of unflatten_constrain.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unflatten_constrain-Union{Tuple{T}, Tuple{ModelWrapper, AbstractVector{T}}} where T<:Real","page":"Home","title":"ModelWrappers.unflatten_constrain","text":"Constrain and Unflatten vector 'θᵤ' given 'model' constraints.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#StatsBase.sample!-Tuple{Random.AbstractRNG, ModelWrapper}","page":"Home","title":"StatsBase.sample!","text":"sample!(_rng, model)\n\n\nInplace version of sample.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#StatsBase.sample-Tuple{Random.AbstractRNG, ModelWrapper}","page":"Home","title":"StatsBase.sample","text":"sample(_rng, model)\n\n\nSample from 'model' prior and return as NamedTuple.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"}]
}
