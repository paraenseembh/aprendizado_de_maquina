/**
 * Escreva a sua solução aqui
 * Code your solution here
 * Escriba su solución aquí


 Liste os nomes de todos os artistas que não possuem nenhum álbum cadastrado no catálogo.

O resultado deve está ordenado pelo nome do artista.


 */

SELECT ar.name 
FROM artists ar 
LEFT JOIN albums al ON ar.id = al.artist_id WHERE al.artist_id IS NULL  ; 