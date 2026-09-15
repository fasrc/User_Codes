using MPI
using LinearAlgebra

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
comm_size = MPI.Comm_size(comm)
root = 0
MPI.Barrier(comm)

N = 1e8;
function count_pi(N)
    R = 1;
    local_count = 0
    for i in 1:N
        coords = rand(2)
        if norm(coords) < R
            local_count = local_count + 1
        end
    end
    return local_count
end
local_count = count_pi(N)
    
MPI.Barrier(comm)
total_count = MPI.Reduce(local_count, +, root, comm)

if rank == root
    print("\n Estimate of pi is: ")
    print(total_count / N / comm_size * 4)
end

MPI.Finalize()