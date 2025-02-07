import numpy as np
from scipy.integrate import dblquad
from fem_config import *

f = lambda u,v,xa,ya,xb,yb,xc,yc: f_0(T(xa,ya,xb,yb,xc,yc,u,v)[0],T(xa,ya,xb,yb,xc,yc,u,v)[1])
c = lambda u,v,xa,ya,xb,yb,xc,yc: c_0(T(xa,ya,xb,yb,xc,yc,u,v)[0],T(xa,ya,xb,yb,xc,yc,u,v)[1])
b1= lambda u,v,xa,ya,xb,yb,xc,yc: b1_0(T(xa,ya,xb,yb,xc,yc,u,v)[0],T(xa,ya,xb,yb,xc,yc,u,v)[1])
b2= lambda u,v,xa,ya,xb,yb,xc,yc: b2_0(T(xa,ya,xb,yb,xc,yc,u,v)[0],T(xa,ya,xb,yb,xc,yc,u,v)[1])
a = lambda u,v,xa,ya,xb,yb,xc,yc: a_0(T(xa,ya,xb,yb,xc,yc,u,v)[0],T(xa,ya,xb,yb,xc,yc,u,v)[1])
p1=lambda u,v: 1-v-u
p2=lambda u,v: u
p3=lambda u,v: v

def T(xa,ya,xb,yb,xc,yc,r,s):
    x=(xa-xb)*r+(xc-xa)*s+xa
    y=(ya-yb)*r+(yc-ya)*s+ya
    return x,y

def integ(f):
    upper_limit_y=lambda v: 1-v
    I=dblquad(f, 0, 1, 0, upper_limit_y)[0]
    return I

def det(xa,ya,xb,yb,xc,yc):
    d=(xa-xb)*(yc-ya)-(xc-xa)*(ya-yb)
    return np.abs(d)

def encontrar_maximos_y_minimos(Tb_ext, Tb_int):
    Tb_ext = [x[0] if type(x)==list else x for x in Tb_ext]
    Tb_int = [x[0] if type(x)==list else x for x in Tb_int]
    Tb_ext = [0 if x is None else x for x in Tb_ext]
    Tb_int = [0 if x is None else x for x in Tb_int]
    if not Tb_ext:
        Tb_ext = [0]
    if not Tb_int:
        Tb_int = [0]
    combinada = Tb_ext + Tb_int
    maximo = max(combinada)
    minimo = max(0, min(combinada))
    return maximo, minimo

def cargar_malla_npz(nombre_archivo):
    print(f"Cargando malla desde {nombre_archivo}...")
    datos = np.load(nombre_archivo, allow_pickle=True)
    malla = {
        "vertices": datos["vertices"],
        "triangles": datos["triangles"],
        "clasificaciones": datos["clasificaciones"].tolist()
    }
    print("Malla cargada correctamente.")
    return malla

def aplicar_dir_ext_V(Tb, P, U, cl):
    U = np.array(U)
    for i, (x, y) in enumerate(P):
        for j, T in enumerate(Tb):
            if cl[i] == "ext" + str(j) and T is not None:
                U[i] = T
    return U

def aplicar_dir_int_V(Tbl, P, U, cl):
    U = np.array(U)
    if len(Tbl)==0:
        return U
    for i, (x, y) in enumerate(P):
        for j, T in enumerate(Tbl):
            if cl[i] == "int" + str(j) and T is not None:
                U[i] = T
    return U

def aplicar_dir_ext_M(Tb, P, M, cl):
    for i, (x, y) in enumerate(P):
        for j, T in enumerate(Tb):
            if cl[i] == "ext" + str(j) and T is not None:
                M[i,:] = 0
                M[:,i] = 0
                M[i,i] = 1
    return M

def aplicar_dir_int_M(Tbl, P, M, cl):
    if len(Tbl)==0:
        return M
    for i, (x, y) in enumerate(P):
        for j, T in enumerate(Tbl):
            if cl[i] == "int" + str(j) and T is not None:
                M[i,:] = 0
                M[:,i] = 0
                M[i,i] = 1
    return M

def Mi(xa,ya,xb,yb,xc,yc):
    if m_0:
        M_local = (det(xa,ya,xb,yb,xc,yc) / 24) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    else:
        M_local=np.zeros([3,3])
    return M_local

def Fi(xa,ya,xb,yb,xc,yc):
  prod1=lambda u,v: f(u,v,xa,ya,xb,yb,xc,yc)*p1(u,v)
  prod2=lambda u,v: f(u,v,xa,ya,xb,yb,xc,yc)*p2(u,v)
  prod3=lambda u,v: f(u,v,xa,ya,xb,yb,xc,yc)*p3(u,v)
  f_local = det(xa,ya,xb,yb,xc,yc) * np.array([integ(prod1), integ(prod2), integ(prod3)])
  return f_local

def Ci(xa,ya,xb,yb,xc,yc):
  prod11=lambda u,v: c(u,v,xa,ya,xb,yb,xc,yc)*p1(u,v)*p1(u,v)
  prod12=lambda u,v: c(u,v,xa,ya,xb,yb,xc,yc)*p1(u,v)*p2(u,v)
  prod13=lambda u,v: c(u,v,xa,ya,xb,yb,xc,yc)*p1(u,v)*p3(u,v)
  prod21=lambda u,v: c(u,v,xa,ya,xb,yb,xc,yc)*p2(u,v)*p1(u,v)
  prod22=lambda u,v: c(u,v,xa,ya,xb,yb,xc,yc)*p2(u,v)*p2(u,v)
  prod23=lambda u,v: c(u,v,xa,ya,xb,yb,xc,yc)*p2(u,v)*p3(u,v)
  prod31=lambda u,v: c(u,v,xa,ya,xb,yb,xc,yc)*p3(u,v)*p1(u,v)
  prod32=lambda u,v: c(u,v,xa,ya,xb,yb,xc,yc)*p3(u,v)*p2(u,v)
  prod33=lambda u,v: c(u,v,xa,ya,xb,yb,xc,yc)*p3(u,v)*p3(u,v)
  c_local = det(xa,ya,xb,yb,xc,yc) * np.array([[integ(prod11), integ(prod12), integ(prod13)], 
                               [integ(prod21), integ(prod22), integ(prod23)], 
                               [integ(prod31), integ(prod32), integ(prod33)]]
                               )
  return c_local

def Bi(xa,ya,xb,yb,xc,yc):
  prod11=lambda u,v: -p1(u,v)*(b1(u,v,xa,ya,xb,yb,xc,yc)+b2(u,v,xa,ya,xb,yb,xc,yc))
  prod12=lambda u,v: p1(u,v)*b1(u,v,xa,ya,xb,yb,xc,yc)
  prod13=lambda u,v: p1(u,v)*b2(u,v,xa,ya,xb,yb,xc,yc)
  prod21=lambda u,v: -p2(u,v)*(b1(u,v,xa,ya,xb,yb,xc,yc)+b2(u,v,xa,ya,xb,yb,xc,yc))
  prod22=lambda u,v: p2(u,v)*b1(u,v,xa,ya,xb,yb,xc,yc)
  prod23=lambda u,v: p2(u,v)*b2(u,v,xa,ya,xb,yb,xc,yc)
  prod31=lambda u,v: -p3(u,v)*(b1(u,v,xa,ya,xb,yb,xc,yc)+b2(u,v,xa,ya,xb,yb,xc,yc))
  prod32=lambda u,v: p3(u,v)*b1(u,v,xa,ya,xb,yb,xc,yc)
  prod33=lambda u,v: p3(u,v)*b2(u,v,xa,ya,xb,yb,xc,yc)
  b_local = det(xa,ya,xb,yb,xc,yc) * np.array([[integ(prod11), integ(prod12), integ(prod13)], 
                               [integ(prod21), integ(prod22), integ(prod23)], 
                               [integ(prod31), integ(prod32), integ(prod33)]]
                               )
  return b_local

def Ai(xa,ya,xb,yb,xc,yc):
  prod11=lambda u,v: 2*a(u,v,xa,ya,xb,yb,xc,yc)
  prod12=lambda u,v: -a(u,v,xa,ya,xb,yb,xc,yc)
  prod13=lambda u,v: -a(u,v,xa,ya,xb,yb,xc,yc)
  prod21=lambda u,v: -a(u,v,xa,ya,xb,yb,xc,yc)
  prod22=lambda u,v: a(u,v,xa,ya,xb,yb,xc,yc)
  prod31=lambda u,v: -a(u,v,xa,ya,xb,yb,xc,yc)
  prod33=lambda u,v: a(u,v,xa,ya,xb,yb,xc,yc)
  a_local = det(xa,ya,xb,yb,xc,yc) * np.array([[integ(prod11), integ(prod12), integ(prod13)], 
                               [integ(prod21), integ(prod22), 0], 
                               [integ(prod31), 0, integ(prod33)]]
                               )
  return a_local