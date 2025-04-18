# CS425 Ricart-Agarwala Algorithm Simulator GUI

A simulator for the Ricart-Agarwala Mutual Exclusion algorithm.

Unfortunately, it makes no guarantees on the correctness of its implementation. It may contain an unholy number of bugs.

The PyQT6 version is more refined and up-to-date. The tkinter version has a complete implementation, but it is not as refined.

## Assumptions

It makes the following assumptions about the distributed system.

- All processing is instantaneous compared to actual message transfer times.
- Time passes in discrete, integer time units.
- No node ever crashes (too difficult to implement well).

## Getting Started

To run the tkinter version:

```bash
uv run main.py
```

To run the PyQT6 version (recommended):

```bash
uv run main_qt.py
```

Note: Most buttons, labels, fields, etc., have hints if you hover over them. They should assist in the navigation of how to use the simulator.

---

Get `uv` here: <https://github.com/astral-sh/uv>

## Screenshots

### Tkinter

<img width="1312" alt="image" src="https://github.com/user-attachments/assets/fb13454d-b4cc-44c4-8dcf-f1adbf9b045f" />

<img width="1268" alt="image" src="https://github.com/user-attachments/assets/11151e4a-df4a-41c6-89b3-6a9aeef5feef" />

<img width="1312" alt="image" src="https://github.com/user-attachments/assets/4cc7b335-5b52-45bd-9821-2401cc983154" />

<img width="1312" alt="image" src="https://github.com/user-attachments/assets/7d08df4d-4e54-40cb-b3e4-036d60366428" />

<img width="1312" alt="image" src="https://github.com/user-attachments/assets/7ab65826-fc5a-4260-a518-267839e2cd68" />

### PyQT6

<img width="1012" alt="image" src="https://github.com/user-attachments/assets/11984e2b-a3d1-4ea1-90ed-9f01db390d86" />

<img width="812" alt="image" src="https://github.com/user-attachments/assets/a97491cc-bf3e-4db1-af08-6eb9f668e443" />

<img width="1012" alt="image" src="https://github.com/user-attachments/assets/9989fc1d-2ddf-4f00-af09-477412b0a5c3" />

<img width="1012" alt="image" src="https://github.com/user-attachments/assets/854b89fc-922b-4a20-b2cb-10870c963d83" />

<img width="1012" alt="image" src="https://github.com/user-attachments/assets/919c5178-4568-4aea-ac0a-6b991fe6baf3" />

## Troubleshooting

### MacOS: `AttributeError: module '_tkinter' has no attribute '__file__'`

See:

- <https://github.com/matplotlib/matplotlib/issues/23074/>
- <https://github.com/astral-sh/python-build-standalone/issues/129>
- <https://github.com/astral-sh/uv/issues/6893>
- <https://github.com/matplotlib/matplotlib/commit/1c02efb55247860b1dc92aa0c5cd5c69b8b8e59d>.

---

At the moment, the only solution is to use a system installed version of Python with the correct tkinter bindings.

If you have an existing system Python install, you can find it with:

```
uv python list
```

and then select it with:

```bash
uv venv --python {version}

#Example
uv venv --python 3.9.22
```

Documentation for these commands can be found at <https://docs.astral.sh/uv/concepts/python-versions/#finding-a-python-executable>.

---

On MacOS, if you have brew installed, you can try running these command to install a compatible version of Python.

```
brew install python@3.9

brew install python-tk@3.9
```

## WSL: `Failed to create wl_display (No such File or directory)`

See:

- <https://github.com/microsoft/WSL/issues/12616>
