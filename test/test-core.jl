############################################################################################
# Checks
############################################################################################

@testset "Core - construct_flatten" begin
    df = FlattenDefault()
    x = [1.]
    constraint = Unconstrained()

    construct_flatten([1.])
    construct_flatten(Unconstrained(), [1.])
    construct_flatten(df, x)
    construct_flatten(df, constraint, x)
    construct_flatten(df, UnflattenStrict(), x)
    construct_flatten(df, UnflattenStrict(), constraint, x)
end

@testset "Core - Param" begin
    Param(_RNG, Normal(), 1.)
    param = Param(Normal(), 1.)
end

@testset "Core - Checkfinite" begin
    @test _checkfinite(.1)
    @test !_checkfinite(Inf)
    @test _checkfinite( (.1, .2, [.2, 3.], zeros(2,3)))

    @test !_checkfinite([.1, -Inf, 3.])
    @test !_checkfinite([[2. 3. ; 4. 5], [.1, -Inf, 3.]])

end

@testset "Core - Checkprior" begin
    con = DistributionConstraint(Distributions.Normal())
    @test _checkprior(con)
    @test !_checkprior(Inf)
    @test _checkprior([con, con])
    @test !_checkprior([Inf, con, con])
    @test !_checkprior([[con, con], [Inf,con, con]])
    @test !_checkprior(
        [[con, con], con, con,
        [Inf, con, con]]
    )
    @test _checkprior([
        [con, con], con, con,
        [con, con]
    ])
    @test !_checkprior([con, con, Fixed(), con])
end

@testset "Core - Checksampleable" begin
    con = DistributionConstraint(Distributions.Normal())
    @test _checksampleable(con)
    @test !_checksampleable(Fixed())
    @test _checksampleable([con, con])
    @test !_checksampleable([Fixed(), con, con])
    @test !_checksampleable([[con, con], [Fixed(), con, con]])
    @test !_checksampleable([[
    [con, con], con, con],
    [Fixed(), con, con]
    ])
    @test _checksampleable([
    [con, con], con, con,
    con, con
    ])
    @test _checksampleable([con con ; con con])
    @test !_checksampleable([con  ; Fixed() ])
end

@testset "Core - Checkparams" begin
    con = DistributionConstraint(Distributions.Normal())
    @test _checkparams(Param(con, 1.))
    @test !_checkparams(Fixed())
end

@testset "Core - _check" begin
    con = DistributionConstraint(Distributions.Normal())
    @test ModelWrappers._check(_RNG, con, .1)
    @test !ModelWrappers._check(_RNG, DistributionConstraint(MvNormal(LinearAlgebra.Diagonal(map(abs2, [1., 2.])))), .1)
end

############################################################################################
# Random
############################################################################################
@testset "Core - sample_constraint" begin
    @test ModelWrappers.sample_constraint(_RNG, Fixed(), nothing) isa Nothing
    @test ModelWrappers.sample_constraint(_RNG, Unconstrained(), 1.0) == 1.0
    @test ModelWrappers.sample_constraint(_RNG, Normal(), 3.) isa Float64
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
    con = DistributionConstraint(Distributions.Normal())
    @test _get_constraint(Param(con, 1.)) isa AbstractConstraint
    @test _get_constraint(Param(Fixed(), 1., )) isa AbstractConstraint
    @test _get_constraint(Param(Unconstrained(), 1., )) isa AbstractConstraint
end

@testset "Core - _get_val" begin
    con = DistributionConstraint(Distributions.Normal())
    @test _get_val(Param(con, 1.)) isa Float64
    @test _get_val(Param(Fixed(), 1.)) isa Float64
    @test _get_val(Param(Unconstrained(), 1.)) isa Float64
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
        (a = Unconstrained(), c = Fixed(), b = [Normal(), Normal()]),
        (a = 1., c = 2., b = [3., 4.]),
    )
    @test _names2 == ["a", "b1", "b2"]
    _names3 = ModelWrappers.paramnames(
        (:a,:c,:b),
        FlattenDefault(),
        (a = Unconstrained(), c = Fixed(), b = [Normal(), Normal()]),
        (a = 1., c = 2., b = [3., 4.]),
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
