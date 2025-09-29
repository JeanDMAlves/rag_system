from utils.db import delete_answer
def plot_iteraction(st, iteraction: tuple, filename: str):
    st.divider()
    st.write(f'Pergunta: {iteraction[1]}')
    st.write(f'Resposta: {iteraction[2]}')
    st.write(f'Data: {iteraction[3]}')
    if st.button('Deletar', key=f'delete_{iteraction[0]}'):
        delete_answer(filename, iteraction[1])
