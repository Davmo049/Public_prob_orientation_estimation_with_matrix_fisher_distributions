import numpy as np

class ParseOffError(Exception):
    def __init__(self, m):
        super().__init__(m)


def write_off(f, points, faces):
    f.write('OFF\n')
    f.write('{} {} 0\n'.format(len(points), len(faces)))
    for p in points:
        f.write('{} {} {}\n'.format(p[0], p[1], p[2]))
    for face in faces:
        f.write('3 {} {} {}\n'.format(face[0], face[1], face[2]))


def parse_off_file(f):
    line = f.readline()
    if line.lower()[:3] != 'off':
        return None, None
    nverts, nfaces, nedges = _parse_off_intro(f)
    verts = []
    for _ in range(nverts):
        verts.append(_parse_off_vertex(f))
    faces = []
    for _ in range(nfaces):
        faces.append(_parse_off_face(f))
    return verts, faces


def _parse_off_intro(f):
    line = f.readline()
    ret = []
    while line[-1] in {'\n', '\r'}:
        line = line[:-1]
    for e in line.split(' '):
        if len(e) != 0 and e.isdigit():
            ret.append(int(e))
    while len(ret) < 3:
        ret.append(0)
    return ret[0], ret[1], ret[2]


def _parse_off_vertex(f):
    line = f.readline()
    ret = []
    while line[-1] in {'\n', '\r'}:
        line = line[:-1]
    for e in line.split(' '):
        if len(e) != 0:
            ret.append(float(e))
    if len(ret) != 3:
        raise ParseOffError("Wrong vertex length")
    return np.array(ret)


def _parse_off_face(f):
    line = f.readline()
    ret = []
    n = None
    while line[-1] in {'\n', '\r'}:
        line = line[:-1]
    for e in line.split(' '):
        if len(e) != 0 and e.isdigit():
            if n is None:
                n = int(e)
            else:
                ret.append(int(e))
    if len(ret) != n:
        raise ParseOffError("Face incorrect")
    return ret
