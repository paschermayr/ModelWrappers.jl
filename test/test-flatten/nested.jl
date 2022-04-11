############################################################################################
# Nested

############################################################################################
@testset "Nested - AbstractArray" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = [1., [2, 3], [4. 5. ; 6. 7.], 8., [9., 10.]]
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64 #Always most common type
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            val = [1., [2, 3], [4. 5. ; 6. 7.], 8., [9., 10.]]
            constraint = [Normal(), MvNormal([1., 1.]), Fixed(), Gamma(), Unconstrained()]
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = unconstrain(_transform, val)
            x_inversetransformed = constrain(_inversetransform, x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            @test x_inversetransformed isa typeof(val)
            val_unconstrained = unconstrain(_transform, val)
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            #!NOTE - Second constraint not correct -> would not be a ModelWrapper
            @test !_check(_RNG, constraint, val)
        end
    end
end

@testset "Nested - AbstractArray - Automatic Differentiation" begin
    val = [1., [2., 3.], [4. 5. ; 6. 7.], 8., [9., 10.]]
    constraint = [Normal(), MvNormal([1., 1.]), Fixed(), Gamma(), Unconstrained()]
    reconstruct = ReConstructor(constraint, val)
    val_flat = flatten(reconstruct, val)

    check_AD = check_AD_closure(constraint, val)
    check_AD(val_flat)
    grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat)
    grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat)
    grad_mod_zy = Zygote.gradient(check_AD, val_flat)[1]
    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_rd = ReverseDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_zy = Zygote.gradient(check_AD, flatten(reconstruct, val))[1]
end

############################################################################################
@testset "Nested - Tuple" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = (1., [2, 3], [4. 5. ; 6. 7.], 8, [9., 10.])
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64 #Always most common type
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            val = (1., [2, 3], [4. 5. ; 6. 7.], 8, [9., 10.])
            constraint = (Normal(), MvNormal([1., 1.]), Fixed(), Fixed(), Unconstrained())
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = unconstrain(_transform, val)
            x_inversetransformed = constrain(_inversetransform, x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            @test x_inversetransformed isa typeof(val)
            val_unconstrained = unconstrain(_transform, val)
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            #!NOTE - Second constraint not correct -> would not be a ModelWrapper
            @test !_check(_RNG, constraint, val)
        end
    end
end

@testset "Nested - Tuple - Automatic Differentiation" begin
    val = (1., [2., 3.], [4. 5. ; 6. 7.], 8., [9., 10.])
    constraint = (Normal(), MvNormal([1., 1.]), Fixed(), Gamma(), Unconstrained())
    reconstruct = ReConstructor(constraint, val)
    val_flat = flatten(reconstruct, val)

    check_AD = check_AD_closure(constraint, val)
    check_AD(val_flat)
    grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat)
    grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat)
    grad_mod_zy = Zygote.gradient(check_AD, val_flat)[1]
    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_rd = ReverseDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_zy = Zygote.gradient(check_AD, flatten(reconstruct, val))[1]
end

############################################################################################
@testset "Nested - NamedTuple" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = (a = Float16(1.0), b = [2, 3], c = [4. 5. ; 6. 7.], d = 8, e = [9., 10.], f = (g = (h = 3.)))
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64 #Always most common type
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            val = (a = Float16(1.0), b = [2, 3], c = [4. 5. ; 6. 7.], d = 8, e = [9., 10.], f = (g = (h = 3.)))
            constraint = (a = Normal(), b = MvNormal([1., 1.]), c = Fixed(), d = Fixed(), e = Unconstrained(), f = (g = (h = Gamma())))
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = unconstrain(_transform, val)
            x_inversetransformed = constrain(_inversetransform, x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            @test x_inversetransformed isa typeof(val)
            val_unconstrained = unconstrain(_transform, val)
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            #!NOTE - Second constraint not correct -> would not be a ModelWrapper
            @test !_check(_RNG, constraint, val)
        end
    end
end

@testset "Nested - NamedTuple - Automatic Differentiation" begin
    val = (a = Float32(1.0), b = [2., 3.], c = [4. 5. ; 6. 7.], d = 8., e = [9., 10.], f = (g = (h = 3.)))
    constraint = (a = Normal(), b = MvNormal([1., 1.]), c = Fixed(), d = Gamma(), e = Unconstrained(), f = (g = (h = Gamma())))
    reconstruct = ReConstructor(constraint, val)
    val_flat = flatten(reconstruct, val)

    check_AD = check_AD_closure(constraint, val)
    check_AD(val_flat)
    grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat)
    grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat)
    grad_mod_zy = Zygote.gradient(check_AD, val_flat)[1]
    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_rd = ReverseDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_zy = Zygote.gradient(check_AD, flatten(reconstruct, val))[1]
end
