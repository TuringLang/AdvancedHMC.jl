using Test, AdvancedHMC

@testset "utilities" begin
    @testset "or!" begin
        @testset "scalar" begin
            @test AdvancedHMC.or!(true, false)
            @test AdvancedHMC.or!(true, true)
            @test AdvancedHMC.or!(false, true)
            @test !AdvancedHMC.or!(false, false)
        end
        @testset "array" begin
            a = [true, true, false, false]
            b = [true, false, true, false]
            cexp = [true, true, true, false]
            c = AdvancedHMC.or!(a, b)
            @test c === a
            @test c == cexp
        end
    end

    @testset "eitheror!" begin
        @testset "nonmutating" begin
            a, b = 1.0, 2.0
            @test AdvancedHMC.eitheror!(a, b, true) == 1.0
            @test AdvancedHMC.eitheror!(a, b, false) == 2.0
            @test AdvancedHMC.eitheror!(4.0, [1.0, 2.0, 3.0], false) == [1.0, 2.0, 3.0]
            @test AdvancedHMC.eitheror!(4.0, [1.0, 2.0, 3.0], true) == [4.0, 4.0, 4.0]
        end

        @testset "mutating" begin
            a, b = [1.0, 2.0, 3.0], 4.0
            c = AdvancedHMC.eitheror!(a, b, true)
            @test a === c
            @test c == [1.0, 2.0, 3.0]

            a, b = [1.0, 2.0, 3.0], 4.0
            c = AdvancedHMC.eitheror!(a, b, false)
            @test a === c
            @test c == fill(4.0, 3)

            a = [1.0 2.0 3.0; 4.0 5.0 6.0]
            b = -[1.0 2.0 3.0; 4.0 5.0 6.0]
            tf = [true; false]
            c = AdvancedHMC.eitheror!(a, b, tf)
            @test a === c
            @test b == -[1.0 2.0 3.0; 4.0 5.0 6.0]
            @test c == [1.0 2.0 3.0; -4.0 -5.0 -6.0]
        end
    end
end
