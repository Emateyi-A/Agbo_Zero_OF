import sympy as sp

# ------------------------
# Root-Finding Methods
# ------------------------
def bisection(f, x, a, b, tol, max_iter):
    prev_mid = None
    table = []
    for i in range(1, max_iter+1):
        mid = (a + b)/2
        f_mid = float(f.subs(x, mid))
        error = abs(mid - prev_mid) if prev_mid is not None else None
        table.append((i, a, b, mid, f_mid, error))
        if f_mid == 0 or (error is not None and error < tol):
            root = mid
            break
        if float(f.subs(x, a)) * f_mid < 0:
            b = mid
        else:
            a = mid
        prev_mid = mid
    if 'root' not in locals():
        root = mid
    return root, error, i, table

def regula_falsi(f, x, a, b, tol, max_iter):
    prev_root = None
    table = []
    for i in range(1, max_iter+1):
        fa = float(f.subs(x, a))
        fb = float(f.subs(x, b))
        r = b - fb*(b - a)/(fb - fa)
        fr = float(f.subs(x, r))
        error = abs(r - prev_root) if prev_root is not None else None
        table.append((i, a, b, r, fr, error))
        if fr == 0 or (error is not None and error < tol):
            root = r
            break
        if fa*fr < 0:
            b = r
        else:
            a = r
        prev_root = r
    if 'root' not in locals():
        root = r
    return root, error, i, table

def secant(f, x, x0, x1, tol, max_iter):
    table = []
    for i in range(1, max_iter+1):
        f0 = float(f.subs(x, x0))
        f1 = float(f.subs(x, x1))
        if f1 - f0 == 0:
            break
        x2 = x1 - f1*(x1 - x0)/(f1 - f0)
        error = abs(x2 - x1)
        table.append((i, x0, x1, x2, float(f.subs(x, x2)), error))
        if error < tol:
            root = x2
            break
        x0, x1 = x1, x2
    if 'root' not in locals():
        root = x2
    return root, error, i, table

def newton_raphson(f, x, x0, tol, max_iter):
    df = sp.diff(f, x)
    table = []
    for i in range(1, max_iter+1):
        f_val = float(f.subs(x, x0))
        df_val = float(df.subs(x, x0))
        if df_val == 0:
            break
        x1 = x0 - f_val/df_val
        error = abs(x1 - x0)
        table.append((i, x0, x1, float(f.subs(x, x1)), error))
        if error < tol:
            root = x1
            break
        x0 = x1
    if 'root' not in locals():
        root = x1
    return root, error, i, table

def fixed_point(f, x, g_expr, x0, tol, max_iter):
    table = []
    g = sp.sympify(g_expr)
    for i in range(1, max_iter+1):
        x1 = float(g.subs(x, x0))
        error = abs(x1 - x0)
        table.append((i, x0, x1, error))
        if error < tol:
            root = x1
            break
        x0 = x1
    if 'root' not in locals():
        root = x1
    return root, error, i, table

def modified_secant(f, x, x0, delta, tol, max_iter):
    table = []
    for i in range(1, max_iter+1):
        f_val = float(f.subs(x, x0))
        f_dx = float(f.subs(x, x0*(1 + delta)))
        if f_dx - f_val == 0:
            break
        x1 = x0 - f_val * delta * x0 / (f_dx - f_val)
        error = abs(x1 - x0)
        table.append((i, x0, x1, float(f.subs(x, x1)), error))
        if error < tol:
            root = x1
            break
        x0 = x1
    if 'root' not in locals():
        root = x1
    return root, error, i, table

# ------------------------
# CLI
# ------------------------
def main():
    print("==== ZOF CLI Root Finder ====")
    eqn = input("Enter equation f(x) (use ** for power, e.g. x**3 - x -2): ")
    x = sp.symbols('x')
    f = sp.sympify(eqn)

    print("Choose method:")
    print("1. Bisection")
    print("2. Regula Falsi")
    print("3. Secant")
    print("4. Newton-Raphson")
    print("5. Fixed Point")
    print("6. Modified Secant")
    choice = int(input("Method number: "))

    tol = float(input("Enter tolerance: "))
    max_iter = int(input("Enter max iterations: "))

    if choice == 1:
        a = float(input("Enter a: "))
        b = float(input("Enter b: "))
        root, error, iters, table = bisection(f, x, a, b, tol, max_iter)
    elif choice == 2:
        a = float(input("Enter a: "))
        b = float(input("Enter b: "))
        root, error, iters, table = regula_falsi(f, x, a, b, tol, max_iter)
    elif choice == 3:
        x0 = float(input("Enter x0: "))
        x1 = float(input("Enter x1: "))
        root, error, iters, table = secant(f, x, x0, x1, tol, max_iter)
    elif choice == 4:
        x0 = float(input("Enter x0: "))
        root, error, iters, table = newton_raphson(f, x, x0, tol, max_iter)
    elif choice == 5:
        x0 = float(input("Enter x0: "))
        g_expr = input("Enter g(x) for Fixed Point: ")
        root, error, iters, table = fixed_point(f, x, g_expr, x0, tol, max_iter)
    elif choice == 6:
        x0 = float(input("Enter x0: "))
        delta = float(input("Enter delta (small number, e.g., 0.01): "))
        root, error, iters, table = modified_secant(f, x, x0, delta, tol, max_iter)
    else:
        print("Invalid choice.")
        return

    print("\n=== Iterations ===")
    for row in table:
        print(row)

    print("\nRoot:", root)
    print("Estimated Error:", error)
    print("Iterations:", iters)


if __name__ == "__main__":
    main()
