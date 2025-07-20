import random
from collections import Counter

CARAMEL_TYPES = ['limon', 'huevo', 'pera']

def generar_jugadores(num_jugadores):
    """Genera jugadores con dos caramelos aleatorios cada uno."""
    return {f'Jugador_{i+1}': [random.choice(CARAMEL_TYPES) for _ in range(2)] for i in range(num_jugadores)}

def contar_dulces(jugadores):
    """Cuenta los caramelos totales de todos los jugadores."""
    todos = []
    for dulces in jugadores.values():
        todos.extend(dulces)
    return Counter(todos)

def formar_chupetines(inventario):
    """Forma chupetines mientras sea posible, devuelve lista y actualiza inventario."""
    chupetines = []
    pasos = []
    count = 1
    while all(inventario[c] >= 2 for c in CARAMEL_TYPES):
        usado = {}
        for dulce in CARAMEL_TYPES:
            inventario[dulce] -= 2
            usado[dulce] = 2
        chupetines.append(usado)
        pasos.append(f"🍭 Chupetín #{count}: {usado}")
        count += 1
    return chupetines, pasos

def generar_comodines(cantidad):
    """Genera una lista de comodines aleatorios."""
    return [random.choice(CARAMEL_TYPES) for _ in range(cantidad)]

def formar_chupetines_con_comodines(inventario, comodines_disponibles):
    """Usa los caramelos restantes + comodines para formar chupetines adicionales."""
    pasos = []
    total_chupetines = 0
    for dulce in comodines_disponibles:
        inventario[dulce] += 1

    while sum(inventario[c] for c in CARAMEL_TYPES) >= 6:
        nuevo = {}
        for dulce in sorted(CARAMEL_TYPES, key=lambda x: inventario[x]):
            usado = min(2, inventario[dulce])
            inventario[dulce] -= usado
            nuevo[dulce] = usado
        total_usado = sum(nuevo.values())
        if total_usado < 6:
            break
        total_chupetines += 1
        pasos.append(f"🎁 Chupetín Extra #{total_chupetines}: {nuevo}")
    return total_chupetines, pasos

def simular_chupetines(num_jugadores=10, mostrar=True):
    """Simula todo el proceso completo de crear chupetines."""
    jugadores = generar_jugadores(num_jugadores)
    inventario = contar_dulces(jugadores)

    if mostrar:
        print("🎲 Caramelos iniciales por jugador:")
        for j, dulces in jugadores.items():
            print(f"  - {j}: {dulces}")

    print("\n📦 Inventario inicial:", dict(inventario))

    # 1. Formar chupetines normales
    chupetines, pasos_normales = formar_chupetines(inventario.copy())

    # 2. Generar comodines por cada chupetín (simula la venta)
    comodines = generar_comodines(len(chupetines) * 2)

    # 3. Formar chupetines adicionales con comodines
    extra_count, pasos_extra = formar_chupetines_con_comodines(inventario, comodines)

    total = len(chupetines) + extra_count

    print("\n✅ RESULTADO FINAL:")
    print(f"🔸 Chupetines normales: {len(chupetines)}")
    print(f"🔸 Chupetines extra (comodines): {extra_count}")
    print(f"🎯 TOTAL: {total} chupetines\n")

    if mostrar:
        print("📜 Detalles:")
        for paso in pasos_normales + pasos_extra:
            print(f"  - {paso}")

    return total

# Ejecutar simulación si se ejecuta como script
if __name__ == "__main__":
    simular_chupetines(12)
