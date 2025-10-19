/**
 * Escreva a sua solução aqui
 * Code your solution here
 * Escriba su solución aquí
 */


SELECT c.nome_cidade, c.regiao , p.qtd_pontos AS media_avaliacoes

FROM Cidades c
JOIN PontosTuristicos p ON c.cidade_id = p.cidade_id

JOIN Avaliacoes a ON a.ponto_id = p.ponto_id


WHERE COUNT(p.nome_ponto) >= 2
ORDER BY AVG(a.nota) AS media_nota;







