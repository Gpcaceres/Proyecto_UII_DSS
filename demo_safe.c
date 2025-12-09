// src/demo_safe.c pruebas
// Ejemplo seguro para probar el pipeline.
// No usa funciones peligrosas como gets, strcpy sin tamaño, system, etc.

#include <stdio.h>
#include <string.h>

int main() {
    char nombre[50];

    printf("Ingresa tu nombre: ");
    fgets(nombre, sizeof(nombre), stdin);  // Seguro: evita desbordamiento

    printf("Hola %s\n", nombre);

    return 0;
}
