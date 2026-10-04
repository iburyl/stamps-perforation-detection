"""Local subpixel edge extraction and robust valley-arc fitting (NumPy/OpenCV)."""
import cv2
import numpy as np


def fit_arc(points, seed, pitch):
    """Fit the inward-facing circular arc, requiring both sides of its apex."""
    origin = np.mean(points, axis=0)
    p = points-origin
    matrix = np.column_stack([2*p[:, 0], 2*p[:, 1], np.ones(len(p))])
    solution = np.linalg.lstsq(matrix, np.sum(p*p, axis=1), rcond=None)[0]
    center = solution[:2]
    radius2 = solution[2]+np.dot(center, center)
    if radius2 <= 0:
        return None
    radius = np.sqrt(radius2)
    for _ in range(12):
        delta = center-p
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
        residual = distance-radius
        scale = max(0.5, 1.4826*np.median(np.abs(residual-np.median(residual))))
        weights = np.minimum(1, 1.5*scale/np.maximum(np.abs(residual), 1e-6))
        jacobian = np.column_stack([delta/distance[:, None], -np.ones(len(p))])
        update = np.linalg.lstsq(jacobian*np.sqrt(weights[:, None]),
                                 -residual*np.sqrt(weights), rcond=None)[0]
        if np.linalg.norm(update) > pitch:
            return None
        center += update[:2]; radius += update[2]
        if np.linalg.norm(update) < 1e-4:
            break
    center += origin
    if not pitch*0.10 < radius < pitch*0.65:
        return None
    residual = np.abs(np.linalg.norm(points-center, axis=1)-radius)
    keep = residual < max(1.5, pitch*0.025)
    if keep.sum() < 9 or keep.mean() < 0.7:
        return None
    pts = points[keep]
    if np.mean(pts[:, 1] > center[1]) < 0.95:
        return None
    apex = center+[0, radius]
    if (abs(apex[0]-seed[0]) > pitch*0.36 or abs(apex[1]-seed[1]) > pitch*0.22 or
            center[0]-pts[:, 0].min() < radius*0.4 or
            pts[:, 0].max()-center[0] < radius*0.4):
        return None
    if abs(np.max(pts[:, 1])-apex[1]) > max(2, pitch*0.045):
        return None
    xx = np.linspace(max(pts[:, 0].min(), center[0]-radius),
                     min(pts[:, 0].max(), center[0]+radius), 45)
    curve = np.column_stack([xx, center[1]+np.sqrt(np.maximum(0, radius**2-(xx-center[0])**2))])
    return {'point': apex, 'curve': curve, 'model': 'circle',
            'rms_px': float(np.sqrt(np.mean(residual[keep]**2))), 'radius_px': float(radius)}


def fit_parabola(points, seed, pitch):
    x = points[:, 0]-seed[0]
    matrix = np.column_stack([x*x, x, np.ones(len(x))])
    weights = np.ones(len(x))
    for _ in range(6):
        coeff = np.linalg.lstsq(matrix*np.sqrt(weights[:, None]), points[:, 1]*np.sqrt(weights), rcond=None)[0]
        residual = points[:, 1]-matrix@coeff
        weights = np.minimum(1, 1.5/np.maximum(np.abs(residual), 1e-6))
    a, b, c = coeff
    if a >= -0.005:
        return None
    vertex = -b/(2*a)
    apex = np.array([seed[0]+vertex, c-b*b/(4*a)])
    keep = np.abs(residual) < max(1.5, pitch*0.025)
    if (keep.sum() < 9 or keep.mean() < 0.8 or
            abs(vertex) > pitch*0.30 or abs(apex[1]-seed[1]) > pitch*0.18 or
            vertex-x.min() < pitch*0.12 or x.max()-vertex < pitch*0.12 or
            -a*min(vertex-x.min(), x.max()-vertex)**2 < 2):
        return None
    xx = np.linspace(x.min(), x.max(), 45)
    return {'point': apex, 'curve': np.column_stack([xx+seed[0], a*xx*xx+b*xx+c]),
            'model': 'parabola', 'rms_px': float(np.sqrt(np.mean(residual[keep]**2)))}


def refine_valleys(signal, positions, depth, seeds, pitch):
    """Use rising intensity/colour transitions near the first-pass boundary."""
    smooth = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 1.2)
    gradient = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)/8
    valid = np.isfinite(depth)
    if valid.sum() < 3:
        return [None]*len(seeds)
    prior = np.interp(positions, positions[valid], depth[valid])
    refined = []
    for seed in seeds:
        fits = []
        for fraction, offset in ((f, o) for f in (0.23, 0.30, 0.37) for o in (-0.12, 0, 0.12)):
            x0 = max(int(positions[0]), int(seed[0]+pitch*(offset-fraction)))
            x1 = min(int(positions[-1]), int(seed[0]+pitch*(offset+fraction)))
            points = []
            for x in range(x0, x1+1):
                initial = prior[x-int(positions[0])]
                if abs(initial-seed[1]) > pitch*0.6:
                    continue
                y0 = max(2, int(initial-pitch*0.10))
                y1 = min(signal.shape[0]-3, int(initial+pitch*0.18))
                if y1 <= y0:
                    continue
                yy = np.arange(y0, y1+1)
                score = gradient[yy, x]*np.exp(-0.5*((yy-initial)/(pitch*0.15))**2)
                y = int(yy[np.argmax(score)])
                if y in (y0, y1) or gradient[y, x] < 1.0:
                    continue
                g0, g1, g2 = gradient[y-1:y+2, x]
                denominator = g0-2*g1+g2
                shift = np.clip(0.5*(g0-g2)/denominator, -0.5, 0.5) if denominator < -1e-6 else 0
                points.append((x, y+shift))
            if len(points) < max(11, (x1-x0)*0.65):
                continue
            points = np.array(points)
            circle = fit_arc(points, seed, pitch)
            parabola = fit_parabola(points, seed, pitch)
            if circle and parabola:
                if np.linalg.norm(circle['point']-parabola['point']) > max(2.5, pitch*0.055):
                    continue
                fits.append(circle)
            elif circle:
                fits.append(circle)
            elif parabola:
                fits.append(parabola)
        # Agreement over different neighbourhood sizes prevents unstable fits.
        stable = [fit for fit in fits if any(other is not fit and
                  np.linalg.norm(fit['point']-other['point']) < max(2, pitch*0.04) for other in fits)]
        refined.append(min(stable, key=lambda fit: (fit['model'] != 'circle', fit['rms_px'])) if stable else None)
    return refined
