############################################################################################
# types

############################################################################################
@testset "Types - Float" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = Float16(1.0)
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == typeof(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            constraint_float = Distributions.Gamma(2,2)
            ReConstructor(constraint_float, val)
            reconstruct = ReConstructor(flatdefault, constraint_float, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == typeof(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint_float, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

            transformer = Transformconstructor(constraint_float, val)

            val_uncon = ModelWrappers.unconstrain(transformer, val)
            val_con = ModelWrappers.constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con


            @test x_inversetransformed isa typeof(val)
            val_unconstrained = unconstrain(_transform, val)
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(x_inversetransformed)
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
        end
    end
end

@testset "Types - Float - Automatic Differentiation" begin
    val = Float32(1.0)
    constraint = Distributions.Gamma(2,2)
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
@testset "Types - Vector Float" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = Float16.([1., 2.])
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)
################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            constraint = Distributions.MvNormal([1.,1.])
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)

################################################################################
# Check if errors if flattened parameter not correct size
            @test_throws ArgumentError flatten(reconstruct, zeros(length(val)+1))
            @test_throws ArgumentError flatten(reconstruct, zeros(length(val)-1))
            @test_throws ArgumentError flattenAD(reconstruct, zeros(length(val)+1))
            @test_throws ArgumentError flattenAD(reconstruct, zeros(length(val)-1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)-1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)-1))

################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

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
            @test val_constrained isa typeof(x_inversetransformed)
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            _check(_RNG, constraint, val)
        end
    end
end

@testset "Types - Vector Float - Automatic Differentiation" begin
    val = [1., 2.]
    constraint = Distributions.MvNormal([1., 1.])
    reconstruct = ReConstructor(constraint, val)
    val_flat = flatten(reconstruct, val)

    check_AD = check_AD_closure(constraint, val)
    check_AD(val_flat)
    grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat)
    grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat)
    grad_mod_zy = Zygote.gradient(check_AD, val_flat)[1]
    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
#    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_rd = ReverseDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_zy = Zygote.gradient(check_AD, flatten(reconstruct, val))[1]
end

############################################################################################
@testset "Types - Array Float" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = Float16.([1. 0.3 ; .3 1.0])
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)

################################################################################
# Check if errors if flattened parameter not correct size
            @test_throws ArgumentError flatten(reconstruct, zeros(3,3))
            @test_throws ArgumentError flatten(reconstruct, zeros(1,1))
            @test_throws ArgumentError flattenAD(reconstruct, zeros(3,3))
            @test_throws ArgumentError flattenAD(reconstruct,  zeros(1,1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)-1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)-1))

################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            constraint = Unconstrained()
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)
################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

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
            @test val_constrained isa typeof(x_inversetransformed)
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            _check(_RNG, constraint, val)
        end
    end
end

@testset "Types - Array Float - Automatic Differentiation" begin
    val = [1. 0.3 ; .3 1.0]
    constraint = Distributions.Distributions.InverseWishart(10., [1. 0. ; 0. 1.])
    reconstruct = ReConstructor(constraint, val)
    val_flat = flatten(reconstruct, val)

    check_AD = check_AD_closure(constraint, val)
    check_AD(val_flat)
    grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat)
    #grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat)
    grad_mod_zy = Zygote.gradient(check_AD, val_flat)[1]
    #@test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(check_AD, flatten(reconstruct, val))
    #grad_mod_rd = ReverseDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_zy = Zygote.gradient(check_AD, flatten(reconstruct, val))[1]
end

############################################################################################
@testset "Types - Integer" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = Int16(1.0)
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            if flattentype isa FlattenAll
                @test eltype(x_flat) == output
            end
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test x_unflatAD2 isa eltype(x_flatAD)
        end
    end
end

############################################################################################
@testset "Types - Array Integer" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = Int16.([1 2 ; 3 4])
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            if flattentype isa FlattenAll
                @test eltype(x_flat) == output
            end
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)
        end
    end
end
