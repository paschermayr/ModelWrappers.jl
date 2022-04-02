############################################################################################
# Checks
############################################################################################

@testset "Core - Checkfinite" begin
    @test _checkfinite(.1)
    @test !_checkfinite(Inf)
    @test _checkfinite( (.1, .2, [.2, 3.], zeros(2,3)))

    @test !_checkfinite([.1, -Inf, 3.])
    @test !_checkfinite([[2. 3. ; 4. 5], [.1, -Inf, 3.]])

end

@testset "Core - Checkprior" begin
    @test _checkprior(Distributions.Normal())
    @test !_checkprior(Inf)
    @test _checkprior([Distributions.Normal(), Distributions.Normal()])
    @test !_checkprior([Inf, Distributions.Normal(), Distributions.Normal()])
    @test !_checkprior([[Distributions.Normal(), Distributions.Normal()], [Inf, Distributions.Normal(), Distributions.Normal()]])
end

@testset "Core - Checksampleable" begin
    @test _checksampleable(Distributions.Normal())
    @test !_checksampleable(Fixed())
    @test _checksampleable([Distributions.Normal(), Distributions.Normal()])
    @test !_checksampleable([Fixed(), Distributions.Normal(), Distributions.Normal()])
    @test !_checksampleable([[Distributions.Normal(), Distributions.Normal()], [Fixed(), Distributions.Normal(), Distributions.Normal()]])
end

@testset "Core - Checkparams" begin
    @test _checkparams(Param(.1, Normal()))
    @test !_checkparams(Fixed())
end

@testset "Core - Checkparam" begin
    @test ModelWrappers._checkparam(_RNG, .1, Normal())
    @test !ModelWrappers._checkparam(_RNG, .1, MvNormal(LinearAlgebra.Diagonal(map(abs2, [1., 2.]))))
end

############################################################################################
# Random
############################################################################################
@testset "Core - sample_constraint" begin
    @test ModelWrappers.sample_constraint(_RNG, nothing) isa Nothing
    @test ModelWrappers.sample_constraint(_RNG, 1.) isa Nothing
    @test ModelWrappers.sample_constraint(_RNG, Normal()) isa Float64
end

@testset "Core - log_prior" begin
    @test ModelWrappers.log_prior(Fixed(), nothing) == 0.0
    @test ModelWrappers.log_prior([Normal(), Normal()], 0.0) == 0.0
    @test ModelWrappers.log_prior([Normal(), Normal()], [0.,0.]) isa Float64
    @test ModelWrappers.log_prior(Normal(), 2.) isa Float64
end

@testset "Core - log_prior_with_transform" begin
    @test ModelWrappers.log_prior_with_transform(Fixed(), nothing) == 0.0
    @test ModelWrappers.log_prior_with_transform([Normal(), Normal()], 0.0) == 0.0
    @test ModelWrappers.log_prior_with_transform([Normal(), Normal()], [0.,0.]) isa Float64
    @test ModelWrappers.log_prior_with_transform(Normal(), 2.) isa Float64
end

@testset "Core - log_abs_det_jac" begin
end

############################################################################################
# Utility
############################################################################################
@testset "Core - _get_constraint" begin
    @test _get_constraint(Param(1., Normal())) isa Distribution
    @test _get_constraint(Param(1., Fixed())) isa AbstractConstraint
    @test _get_constraint(Param(1., Unconstrained())) isa AbstractConstraint
end

@testset "Core - _get_val" begin
    @test _get_val(Param(1., Normal())) isa Float64
    @test _get_val(Param(1., Fixed())) isa Float64
    @test _get_val(Param(1., Unconstrained())) isa Float64
end

@testset "Core - _allparam" begin
end

@testset "Core - _anyparam" begin
end

@testset "Core - _paramnames" begin
    @test length(ModelWrappers._paramnames(:a, 1)) == 1
    @test length(ModelWrappers._paramnames(:a, 2)) == 2
    _names = ModelWrappers._paramnames((:a, :b), (1,2))
    @test _names == ["a", "b1", "b2"]
end

@testset "Core - paramnames" begin
    _names2 = ModelWrappers.paramnames(
        FlattenDefault(),
        (a = 1., c = 2., b = [3., 4.]),
        (a = Unconstrained(), c = Fixed(), b = [Normal(), Normal()])
    )
    @test _names2 == ["a", "b1", "b2"]
    _names3 = ModelWrappers.paramnames(
        (:a,:c,:b),
        FlattenDefault(),
        (a = 1., c = 2., b = [3., 4.]),
        (a = Unconstrained(), c = Fixed(), b = [Normal(), Normal()])
    )
    @test all(_names2 .== _names3)
end

@testset "Core - paramcount" begin
    _vals = ModelWrappers.paramcount(
        FlattenDefault(),
        (a = 1., c = 2., b = [3., 4.])
    )
    @test _vals == (1,1,2)
    _vals2 = ModelWrappers.paramcount(
        (:a,:c,:b),
        FlattenDefault(),
        (a = 1., c = 2., b = [3., 4.])
    )
    @test all(_vals .== _vals2)
end
