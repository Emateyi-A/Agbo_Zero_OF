from flask import Flask, render_template, request
import sympy as sp

app = Flask(__name__)

# --------------------------
# Root-Finding Methods
# --------------------------

def bisection(f, x, a, b, tol, max_iter):
    prev_mid = None
    table = []
    root = None
    error = None
    for i in range(1, max_iter + 1):
        mid = (a + b) / 2
        f_mid = float(f.subs(x, mid))
        error = abs(mid - prev_mid) if prev_mid is not None else None
        table.append({"iter": i, "a": a, "b": b, "mid": mid, "f_mid": f_mid, "error": error})
        if f_mid == 0 or (error is not None and error < tol):
            root = mid
            break
        if float(f.subs(x, a)) * f_mid < 0:
            b = mid
        else:
            a = mid
        prev_mid = mid
    if root is None:
        root = mid
    return root, error, i, table

def regula_falsi(f, x, a, b, tol, max_iter):
    prev_root = None
    table = []
    root = None
    error = None
    for i in range(1, max_iter + 1):
        fa = float(f.subs(x, a))
        fb = float(f.subs(x, b))
        r = b - fb*(b - a)/(fb - fa)
        f_r = float(f.subs(x, r))
        error = abs(r - prev_root) if prev_root is not None else None
        table.append({"iter": i, "a": a, "b": b, "root": r, "f_root": f_r, "error": error})
        if f_r == 0 or (error is not None and error < tol):
            root = r
            break
        if fa * f_r < 0:
            b = r
        else:
            a = r
        prev_root = r
    if root is None:
        root = r
    return root, error, i, table

def secant(f, x, x0, x1, tol, max_iter):
    table = []
    root = None
    error = None
    for i in range(1, max_iter+1):
        f0 = float(f.subs(x, x0))
        f1 = float(f.subs(x, x1))
        if f1 - f0 == 0:
            break
        x2 = x1 - f1*(x1 - x0)/(f1 - f0)
        error = abs(x2 - x1)
        table.append({"iter": i, "x0": x0, "x1": x1, "x2": x2, "f_x2": float(f.subs(x, x2)), "error": error})
        if error < tol:
            root = x2
            break
        x0, x1 = x1, x2
    if root is None:
        root = x2
    return root, error, i, table

def newton_raphson(f, x, x0, tol, max_iter):
    f_prime = sp.diff(f, x)
    table = []
    root = None
    error = None
    for i in range(1, max_iter+1):
        f0 = float(f.subs(x, x0))
        f0_prime = float(f_prime.subs(x, x0))
        if f0_prime == 0:
            break
        x1 = x0 - f0/f0_prime
        error = abs(x1 - x0)
        table.append({"iter": i, "x0": x0, "x1": x1, "f_x1": float(f.subs(x, x1)), "error": error})
        if error < tol:
            root = x1
            break
        x0 = x1
    if root is None:
        root = x1
    return root, error, i, table

def fixed_point(x0, g_expr, x_sym, tol, max_iter):
    g = sp.sympify(g_expr)
    table = []
    root = None
    error = None
    for i in range(1, max_iter+1):
        x1 = float(g.subs(x_sym, x0))
        error = abs(x1 - x0)
        table.append({"iter": i, "x0": x0, "x1": x1, "error": error})
        if error < tol:
            root = x1
            break
        x0 = x1
    if root is None:
        root = x1
    return root, error, i, table

def modified_secant(f, x, x0, delta, tol, max_iter):
    table = []
    root = None
    error = None
    for i in range(1, max_iter+1):
        f0 = float(f.subs(x, x0))
        f_delta = float(f.subs(x, x0*(1 + delta)))
        if f_delta - f0 == 0:
            break
        x1 = x0 - f0 * delta * x0 / (f_delta - f0)
        error = abs(x1 - x0)
        table.append({"iter": i, "x0": x0, "x1": x1, "f_x1": float(f.subs(x, x1)), "error": error})
        if error < tol:
            root = x1
            break
        x0 = x1
    if root is None:
        root = x1
    return root, error, i, table

# --------------------------
# Flask Routes
# --------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    table = None
    method = None

    if request.method == "POST":
        eqn = request.form["equation"]
        method = request.form["method"]
        tol = float(request.form["tol"])
        max_iter = int(request.form["max_iter"])
        x = sp.symbols('x')
        f = sp.sympify(eqn)

        if method == "bisection":
            a = float(request.form["a"])
            b = float(request.form["b"])
            root, error, iterations, table = bisection(f, x, a, b, tol, max_iter)

        elif method == "regula_falsi":
            a = float(request.form["a"])
            b = float(request.form["b"])
            root, error, iterations, table = regula_falsi(f, x, a, b, tol, max_iter)

        elif method == "secant":
            x0 = float(request.form["x0"])
            x1 = float(request.form["x1"])
            root, error, iterations, table = secant(f, x, x0, x1, tol, max_iter)

        elif method == "newton_raphson":
            x0 = float(request.form["x0"])
            root, error, iterations, table = newton_raphson(f, x, x0, tol, max_iter)

        elif method == "fixed_point":
            x0 = float(request.form["x0"])
            g_expr = request.form["g_expr"]
            root, error, iterations, table = fixed_point(x0, g_expr, x, tol, max_iter)

        elif method == "modified_secant":
            x0 = float(request.form["x0"])
            delta = float(request.form["delta"])
            root, error, iterations, table = modified_secant(f, x, x0, delta, tol, max_iter)

        result = {"root": root, "error": error, "iterations": iterations}

    return render_template("index.html", result=result, table=table, method=method)


if __name__ == "__main__":
    app.run(debug=True)
