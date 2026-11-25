import numpy as np
from scipy.integrate import quad

# Constantes

pi = np.pi
c = 3e8 # m/s
u0 = 4*pi*1e-7
e0 = 1/(36*pi) * 1e-9

# Funções básicas 

def fatorial(n):
    if n == 0:
        return 1
    else:
        produto = 1
        for k in range(1, n + 1, 1):
            produto *= k
        return produto

# Função Gamma

def integrando_funcao_Gamma(x, z):
    exp = np.exp(-x)
    polinomio = np.power(x, z - 1)
    return polinomio * exp

def funcao_Gamma(Z):
    if np.isscalar(Z):
        Z = np.array([Z], dtype=float)
    resultados = []
    for z in Z:
        # Polos da função Gamma (0, -1, -2, ...)
        if z <= 0 and np.isclose(z, np.round(z), rtol=0, atol=1e-12):
            resultados.append(np.inf)
            continue

        # Desloca z para o intervalo [1, 2) onde a integral é suave e bem comportada, aproveitando a propriedade \Gamma(z+1) = z\Gamma(z)
        multiplicador = 1.0
        while z < 1:
            multiplicador /= z
            z += 1
        while z >= 2:
            z -= 1
            multiplicador *= z

        def f(x):
            return integrando_funcao_Gamma(x, z)
        resultado, erro = quad(f, 0, np.inf)
        resultados.append(resultado * multiplicador)

    resultados = np.array(resultados)
    if len(resultados) == 1:
        return resultados[0]
    return resultados

# Funções de Bessel

def J(t, alpha, M=75):
    if not (np.isscalar(alpha)):
        raise ValueError('\u03B1 deve ser um escalar')
    if (np.isclose(alpha, np.round(alpha), rtol=0, atol=1e-12)) and (alpha < 0):
        return (-1)**(int(-np.round(alpha)))*J(t, -np.round(alpha), M)
    soma = 0
    for m in range(0, M + 1, 1):
        if m == 0:
            denominador2 = funcao_Gamma(alpha + 1)
        else:
            denominador2 = (m+alpha)*denominador2
        
        # Se Gamma divergir (polo), 1/Gamma = 0; pula o termo para evitar NaN em t = 0
        if np.isinf(denominador2):
            continue

        sinal = (-1)**m
        denominador1 = fatorial(m)
        denominador = denominador1 * denominador2

        numerador = np.power(t / 2, 2 * m + alpha)

        soma += sinal * numerador / denominador

    return soma

def I(t, alpha, M=75):
    if not (np.isscalar(alpha)):
        raise ValueError('\u03B1 deve ser um escalar')
    if (np.isclose(alpha, np.round(alpha), rtol=0, atol=1e-12)) and (alpha < 0):
        return I(t, -np.round(alpha), M)
    soma = 0
    for m in range(0, M + 1, 1):
        if m == 0:
            denominador2 = funcao_Gamma(m + alpha + 1)
        else:
            denominador2 = (m+alpha)*denominador2
        
        # Se Gamma divergir (polo), 1/Gamma = 0; pula o termo para evitar NaN em t = 0
        if np.isinf(denominador2):
            continue

        denominador1 = fatorial(m)
        denominador = denominador1 * denominador2

        numerador = np.power(t / 2, 2 * m + alpha)

        soma += numerador / denominador

    return soma

def Y(t, alpha, M = 75):
    if not (np.isscalar(alpha)):
        raise ValueError('\u03B1 deve ser um escalar')

    # Se alpha for inteiro, aproxima o limite numericamente para evitar 0/0
    if np.isclose(alpha, np.round(alpha), rtol = 0, atol=1e-12):
        e = 1e-6
        return 0.5 * (Y(t,alpha+e,M) + Y(t, alpha-e,M))

    num = J(t,alpha,M)*np.cos(alpha*pi) - J(t,-alpha,M)
    den = np.sin(alpha*pi)
    return num / den

def K(t, alpha, M = 75):
    if not (np.isscalar(alpha)):
        raise ValueError('\u03B1 deve ser um escalar')
    if alpha < 0:
        return K(t, -alpha, M)

    # Se alpha for inteiro, aproxima o limite numericamente para evitar 0/0
    if np.isclose(alpha, np.round(alpha), rtol = 0, atol=1e-12):
        e = 1e-6
        return 0.5 * (K(t,alpha+e,M) + K(t, alpha-e,M))

    num = pi*(I(t,-alpha,M) - I(t,alpha,M))
    den = 2*np.sin(alpha*pi)
    return num / den

# Derivadas das funções de Bessel

def DJ(t, alpha, M = 75):
    Ja_m1 = J(t, alpha-1, M)
    Ja_p1 = J(t, alpha+1, M)
    return 0.5*(Ja_m1-Ja_p1)

def DI(t, alpha, M = 75):
    Ia_m1 = I(t, alpha-1, M)
    Ia_p1 = I(t, alpha+1, M)
    return 0.5*(Ia_m1+Ia_p1)

def DY(t, alpha, M = 75):
    Ya_m1 = Y(t, alpha-1, M)
    Ya_p1 = Y(t, alpha+1, M)
    return 0.5*(Ya_m1-Ya_p1)

def DK(t, alpha, M = 75):
    Ka_m1 = K(t, alpha-1, M)
    Ka_p1 = K(t, alpha+1, M)
    return -0.5*(Ka_m1+Ka_p1)

# Algoritmos auxiliares

def p(b, n1, k0):
    p2 = (n1*k0)**2 - b**2
    return np.sqrt(p2)
def q(b, n2, k0):
    q2 = b**2 - (n2*k0)**2
    return np.sqrt(q2)

def F(b, m, a, n1, n2, lambda0, M =80):
    k0 = 2*pi/lambda0

    pa = p(b, n1, k0)*a
    qa = q(b, n2, k0)*a

    J_ = J(pa, m, M)
    dJ_ = DJ(pa, m, M)
    K_ = K(qa, m, M)
    dK_ = DK(qa, m, M)

    P1 = dJ_/(pa*J_) + dK_/(qa*K_)
    P2 = dJ_/(pa*J_) + ((n2/n1)**2)*dK_/(qa*K_)

    return P1*P2 - (m**2)*(1/(pa**2) + 1/(qa**2))*(1/(pa**2) + ((n2/n1)**2)*1/(qa**2))

def localizar_raizes(x, y, e=1e-5):
    raizes = []
    if len(x) != len(y):
        raise ValueError('os vetores x e y devem ter o mesmo tamanho.')
    if len(x) == 0:
        raise ValueError('Os vetores devem conter elementos.')
    N = len(x)

    possiveis_raizes = []
    for k in range(0,N-1,1):
        if y[k] == 0:
            raizes.append(x[k])
            possiveis_raizes = []
            continue
        elif y[k]*y[k+1] < 0:
            raiz = x[k] - y[k]*(x[k+1]-x[k])/(y[k+1]-y[k])
            raizes.append(raiz)
            possiveis_raizes = []
            continue
        elif abs(y[k]) < e:
            possiveis_raizes.append(k)
            if abs(y[k+1]) > e:
                i = np.argmin(np.abs(y[possiveis_raizes]))
                raiz = x[possiveis_raizes[i]]
                if not raiz in raizes: 
                    raizes.append(raiz)
                possiveis_raizes = []
    if np.abs(y[-1]) < e:
        raizes.append(x[-1])
    return np.array(raizes)

def retornar_raiz(b_array, m, a, n1, n2, lambda0, M = 80, e=1e-5):
    k0 = 2*pi/lambda0
    if (np.min(b_array/k0) <= n2) or (np.max(b_array/k0) >= n1):
        raise ValueError('os limites de b_array devem estar no intervalo aberto entre n2 e n1.')
    F_array = F(b_array, m, a, n1, n2, lambda0, M)
    mask = np.abs(F_array) < 2
    F_array = F_array[mask]
    raizes = localizar_raizes(b_array[mask]/k0, F_array, e)
    return raizes
