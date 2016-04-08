

# constrained maximization exercises

## portfolio choice problem

module HW_constrained

	using JuMP, NLopt, DataFrames, Ipopt, Distances

	export data, table_NLopt, table_JuMP

	function data(a=0.5)

		price = [1.0, 1.0, 1.0]
		endowment = [2.0, 0.0, 0.0]
		z1 = [1.0, 1.0, 1.0, 1.0]
		z2 = [0.72, 0.92, 1.12, 1.32]
		z3 = [0.86, 0.96, 1.06, 1.16]
		truth = DataFrame(a = [0.5,1.0,5.0], c = [1.00801,1.00401,1.0008], omega1 = [-1.41237,-0.206197,0.758762], omega2 =[0.801458,0.400729,0.0801456], omega3 = [1.60291,0.801462,0.160291], fval = [-1.20821,-0.732819,-0.013422])

		return Dict("price" => price, "endowment" => endowment, "z1" => z1, "z2" => z2, "z3" => z3, "a" => a, "truth" => truth)

	end


	function max_JuMP(a=0.5)

		m = Model()

	     @defVar(m, c)
	     @defVar(m, w1)
			 @defVar(m, w2)
			 @defVar(m, w3)

			 z2 = [0.72, 0.92, 1.12, 1.32]
			 z3 = [0.86, 0.96, 1.06, 1.16]

	     @setNLObjective(m, Min, exp(-a*c) + 1/16*sum{exp(-a*(w1 + w2*z2[rem(s+3,4)+1] + w3*z3[div(s-1,4)+1])), s=1:16})
	     @addNLConstraint(m,c + w1 + w2 + w3 <= 2)

	     solve(m)

			 return Dict("c" => getValue(c), "w1"=> getValue(w1), "w2"=> getValue(w2), "w3"=> getValue(w3), "fval"=> getObjectiveValue(m))

	end

	function table_JuMP()

		a = [0.5,1.0,5.0]
		max1 = max_JuMP(a[1])
		max2 = max_JuMP(a[2])
		max3 = max_JuMP(a[3])
		fval = -[max1["fval"],max2["fval"],max3["fval"]]

		c = [max1["c"],max2["c"],max3["c"]]
		omega1 = [max1["w1"],max2["w1"],max3["w1"]]
		omega2 = [max1["w2"],max2["w2"],max3["w2"]]
		omega3 = [max1["w3"],max2["w3"],max3["w3"]]

		return DataFrame(a=a, c=c, omega1 =omega1, omega2 = omega2, omega3 = omega3, fval = fval)

	end

	function obj(x::Vector,grad::Vector,data::Dict)

		# Get the data
		z1 = data["z1"]
		z2 = data["z2"]
		z3 = data["z3"]
		a = data["a"]

		# Define the utility function and it's derivative
		u(c) = -exp(-a*c)
		v(c) = a*exp(-a*c)

		# Define the gradients
		if length(grad) > 0
      grad[1] = -v(x[1])
			# Derivatives with respect to the omegas
			y2 = 0
			y3 = 0
			y4 = 0
				for i in 1:4
					for j in 1:4
						y2 = y2 + v(x[2]+z2[i]*x[3]+z3[j]*x[4])/16
						y3 = y3 + z2[i]*v(x[2]+z2[i]*x[3]+z3[j]*x[4])/16
						y4 = y4 + z3[j]*v(x[2]+z2[i]*x[3]+z3[j]*x[4])/16
					end
				end
				grad[2] = -y2
				grad[3] = -y3
				grad[4] = -y4
		end

		# Part of the objective function
		y = 0
		for i in 1:4
			for j in 1:4
				y = y + u(x[2]+z2[i]*x[3]+z3[j]*x[4])/16
			end
		end

    return -(u(x[1])+y)
	end


	function constr(x::Vector,grad::Vector,data::Dict)

		# Get data
		price = data["price"]
		endowment = data["endowment"]
		if length(grad) > 0
	 		grad[1] = 1
			grad[2] = price[1]
	 		grad[3] = price[2]
			grad[4] = price[3]
		end
	 	return x[1] + dot(price,x[2:4]-endowment)

	end

	function max_NLopt(a=0.5)

		# Define an Opt object: algorithm and dimensions of choice
		opt = Opt(:LD_MMA, 4)

		# Set bounds and tolerance
		lower_bounds!(opt, [0,-Inf,-Inf,-Inf])
		xtol_rel!(opt,1e-6)

		# Define objective function
		min_objective!(opt, (x,g) -> obj(x,g,data(a)))

		# Define constraint
		inequality_constraint!(opt, (x,g) -> constr(x,g,data(a)), 1e-8)

		# Call optimize
		(minf,minx,ret) = optimize(opt, [0., 0.,0.,0.])

	end

	function table_NLopt()

		a = [0.5,1.0,5.0]
		max1 = max_NLopt(a[1])
		max2 = max_NLopt(a[2])
		max3 = max_NLopt(a[3])
		fval = -[max1[1],max2[1],max3[1]]
		c = [max1[2][1],max2[2][1],max3[2][1]]
		omega1 = [max1[2][2],max2[2][2],max3[2][2]]
		omega2 = [max1[2][3],max2[2][3],max3[2][3]]
		omega3 = [max1[2][4],max2[2][4],max3[2][4]]

		return DataFrame(a=a, c=c, omega1 =omega1, omega2 = omega2, omega3 = omega3, fval = fval)

	end

	# function `f` is for the NLopt interface, i.e.
	# it has 2 arguments `x` and `grad`, where `grad` is
	# modified in place
	# if you want to call `f` with more than those 2 args, you need to
	# specify an anonymous function as in
	# other_arg = 3.3
	# test_finite_diff((x,g)->f(x,g,other_arg), x )
	# this function cycles through all dimensions of `f` and applies
	# the finite differencing to each. it prints some nice output.
	function test_finite_diff(f::Function,x::Vector{Float64},tol=1e-6)

		grad = zeros(4)

		if euclidean(grad, finite_diff(x->obj(x, grad, data()),x)) > tol
			println("Differentiation test fails")
			else println("Differentiation test passed!")
		end
		return euclidean(grad, finite_diff(x->obj(x, grad, data()),x))

	end

	# do this for each dimension of x
	# low-level function doing the actual finite difference
	function finite_diff(f::Function,x::Vector)

		derivative = []
		y = zeros(length(x))


		for i in 1:length(x)
			y[:] = x[:]
			y[i] += sqrt(eps())
			push!(derivative, (f(y) - f(x))/sqrt(eps()))
		end

		return derivative

	end

	function runAll()
		println("running tests:")
		include("test/runtests.jl")
		println("")
		println("JumP:")
		print(table_JuMP())
		println("")
		println("NLopt:")
		print(table_NLopt())
		println("")

	end


end
