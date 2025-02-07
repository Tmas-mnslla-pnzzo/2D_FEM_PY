import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
import sys
from fem_funciones import *

if len(sys.argv) < 2:
    print("Error: Debes proporcionar el nombre del archivo.")
    sys.exit(1) 

nombre = sys.argv[1]

malla = cargar_malla_npz(nombre + ".npz")
cl = malla["clasificaciones"]
trip = malla["triangles"]
P = malla["vertices"]
N = len(P)
per=len(trip)
unidades=["Temperatura","K°"]

print("Ensamblando matrices globales...")
M = lil_matrix((N, N))
F = np.zeros(N)
C = lil_matrix((N, N))
B = lil_matrix((N, N))
A = lil_matrix((N, N))

count=0
for ele in trip:
    x1,y1=P[ele[0]]
    x2,y2=P[ele[1]]
    x3,y3=P[ele[2]]

    mi = Mi(x1,y1,x2,y2,x3,y3)
    fi = Fi(x1,y1,x2,y2,x3,y3)
    ci = Ci(x1,y1,x2,y2,x3,y3)
    bi = Bi(x1,y1,x2,y2,x3,y3)
    ai = Ai(x1,y1,x2,y2,x3,y3)
    for a_i in range(3):
        F[ele[a_i]] += fi[a_i]
        for b_i in range(3):
            M[ele[a_i], ele[b_i]] += mi[a_i, b_i]
            C[ele[a_i], ele[b_i]] += ci[a_i, b_i]
            B[ele[a_i], ele[b_i]] += bi[a_i, b_i]
            A[ele[a_i], ele[b_i]] += ai[a_i, b_i]
    print("Elementos cargados al "+str(round(count*(100/per),2))+"%")
    count+=1

M = csr_matrix(M)
C = csr_matrix(C)
B = csr_matrix(B)
A = csr_matrix(A)
G=A+B+C
print("Matrices globales ensambladas correctamente.")

F=aplicar_dir_ext_V(cond_dir_ext, P, F, cl)
F=aplicar_dir_int_V(cond_dir_int, P, F, cl)

print("Resolviendo la ecuación de convección-difusión...")

if m_0:
    k = Tm / tn
    Gt=M + k*G
    U = np.zeros(N)
    Gt=aplicar_dir_ext_M(cond_dir_ext, P, Gt, cl)
    Gt=aplicar_dir_int_M(cond_dir_int, P, Gt, cl)

    #U = aplicar_init_ext_V(cond_init_ext, P, U, cl)
    #U = aplicar_init_int_V(cond_init_int, P, U, cl)

    Ft=np.dot(M,U)
    Ft=F*k+Ft
    Ft = aplicar_dir_ext_V(cond_dir_ext, P, Ft, cl)
    Ft = aplicar_dir_int_V(cond_dir_int, P, Ft, cl)
    T = np.linspace(0, Tm, tn)
    for t in T:
        U = lsqr(Gt, Ft)[0]
        Ft=np.dot(M,U)
        Ft=F*k+Ft
        Ft = aplicar_dir_ext_V(cond_dir_ext, P, Ft, cl)
        Ft = aplicar_dir_int_V(cond_dir_int, P, Ft, cl)
else:
    G=aplicar_dir_ext_M(cond_dir_int, P, G, cl)
    G=aplicar_dir_int_M(cond_dir_int, P, G, cl)
    U = lsqr(G, F)[0]

print("Solución de la ecuación de convección-difusión completada.")

print("Generando gráfica de la solución...")

x = P[:, 0]
y = P[:, 1]
z = U

plt.tricontourf(x, y, trip, z, levels=50, cmap='plasma')
plt.triplot(x, y, trip, color='black', linewidth=0.5)
plt.gca().set_aspect("equal")
plt.colorbar(label=unidades[0]+" ["+unidades[1]+"]")
if m_0:
    plt.title(f"Solución de la ecuación de convección-difusión con elementos finitos (t = {Tm} s)")
else:
    plt.title(f"Solución de la ecuación de convección-difusión con elementos finitos")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()