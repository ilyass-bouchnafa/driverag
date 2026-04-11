from src.retrieval.query_processor import generate_multi_queries
from src.retrieval.query_processor import generate_hyde_document

# # Question de test pour générer des reformulations multi-query
# question = "Comment trouver sa maison?"

# # Appel de la fonction de génération de requêtes multiples
# result = generate_multi_queries(question)

# # Afficher les queries générées (original + reformulations)
# print("🔄 Queries générées :")
# for q in result:
#     print(f"- {q}")

# Question de test pour HyDE
question = "How does Retrieval-Augmented Generation (RAG) work?"

# Appel de la fonction HyDE
hyde_document = generate_hyde_document(question)

# Affichage du résultat
print("=" * 80)
print("📄 DOCUMENT HYPOTHÉTIQUE GÉNÉRÉ PAR HyDE")
print("=" * 80)
print(hyde_document)
print("=" * 80)
print(f"Longueur : {len(hyde_document)} caractères | {len(hyde_document.split())} mots")