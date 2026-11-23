import numpy as np
from scipy.integrate import quad

# Constantes 

pi = np.pi

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

def J(t, alpha, M=50):
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

def I(t, alpha, M=50):
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

def Y(t, alpha, M = 50):
    if not (np.isscalar(alpha)):
        raise ValueError('\u03B1 deve ser um escalar')

    # Se alpha for inteiro, aproxima o limite numericamente para evitar 0/0
    if np.isclose(alpha, np.round(alpha), rtol = 0, atol=1e-12):
        e = 1e-6
        return 0.5 * (Y(t,alpha+e,M) + Y(t, alpha-e,M))

    num = J(t,alpha,M)*np.cos(alpha*pi) - J(t,-alpha,M)
    den = np.sin(alpha*pi)
    return num / den

def K(t, alpha, M = 50):
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

def DJ(t, alpha, M = 50):
    Ja_m1 = J(t, alpha-1, M)
    Ja_p1 = J(t, alpha+1, M)
    return 0.5*(Ja_m1-Ja_p1)

def DI(t, alpha, M = 50):
    Ia_m1 = I(t, alpha-1, M)
    Ia_p1 = I(t, alpha+1, M)
    return 0.5*(Ia_m1+Ia_p1)

def DY(t, alpha, M = 50):
    Ya_m1 = Y(t, alpha-1, M)
    Ya_p1 = Y(t, alpha+1, M)
    return 0.5*(Ya_m1-Ya_p1)

def DK(t, alpha, M = 50):
    Ka_m1 = K(t, alpha-1, M)
    Ka_p1 = K(t, alpha+1, M)
    return -0.5*(Ka_m1+Ka_p1)