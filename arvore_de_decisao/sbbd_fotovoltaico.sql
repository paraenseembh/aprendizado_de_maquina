/**
 * Escreva a sua solução aqui
 * Code your solution here
 * Escriba su solución aquí



 Um engenheiro de projetos de energia solar precisa criar propostas para clientes com uma exigência específica:
  a montagem do sistema deve usar apenas painéis de um único modelo para atingir exatamente
 a potência máxima de um inversor. Isso evita desperdício de capacidade e simplifica a instalação.



O engenheiro precisa de uma lista de todas as combinações "perfeitas" possíveis entre 
os inversores e os painéis disponíveis em estoque. A lista deve mostrar quais painéis podem 
ser pareados com quais inversores para atingir essa correspondência exata de potência. 
O resultado deve está ordenado pelo id do inversor e o id do painel solar.
 */


SELECT i.inverter_model, i.max_power_watts as inverter_max_power FROM Inverter i ;
    CASE 
        WHEN MOD(inverter_max_power/p.power_watts) = 0 THEN inverter_max_power/p.power_watts
    END as number_of_panels
    ;
    CASE 
        WHEN MOD(inverter_max_power/p.power_watts) = 0 THEN inverter_max_power/p.power_watts
    END as number_of_panels
    ;