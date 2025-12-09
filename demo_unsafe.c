// src/demo_unsafe.c
// Ejemplo deliberadamente inseguro para probar detección del modelo.
// Contiene patrones clásicos de vulnerabilidad buffer overflow y command injection.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    char buffer[10];
    char comando[100];

    printf("Ingresa texto (esto causará overflow): ");
    gets(buffer);   // ⚠ VULNERABLE: gets NO controla tamaño

    printf("Ingresa un comando del sistema: ");
    scanf("%s", comando);

    // ⚠ VULNERABLE: ejecutar entrada del usuario en el sistema
    system(comando);

    // ⚠ strcpy sin control de tamaño
    char destino[5];
    strcpy(destino, buffer);

    return 0;
}
