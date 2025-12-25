# Chunking Approaches: Visual Comparison

## Example Code (2,847 chars / ~570 tokens / 78 lines)

```python
# file: user_service.py
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class User:
    """Represents a user in the system."""
    id: int
    email: str
    name: str
    created_at: datetime
    is_active: bool = True

class UserService:
    """Service for managing user operations.

    This service handles user creation, authentication,
    and profile management with caching support.
    """

    def __init__(self, db_connection, cache_client=None):
        """Initialize the user service.

        Args:
            db_connection: Database connection instance
            cache_client: Optional Redis cache client
        """
        self.db = db_connection
        self.cache = cache_client
        self._user_cache = {}
        logger.info("UserService initialized")

    def create_user(self, email: str, name: str, password: str) -> User:
        """Create a new user account.

        Args:
            email: User's email address (must be unique)
            name: User's display name
            password: Plain text password (will be hashed)

        Returns:
            User: The created user object

        Raises:
            ValueError: If email already exists
            ValidationError: If email format is invalid
        """
        # Validate email format
        if not self._validate_email(email):
            raise ValueError(f"Invalid email format: {email}")

        # Check for existing user
        existing = self.db.query("SELECT id FROM users WHERE email = ?", email)
        if existing:
            raise ValueError(f"User with email {email} already exists")

        # Hash password and create user
        password_hash = self._hash_password(password)
        user_id = self.db.execute(
            "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            email, name, password_hash, datetime.now()
        )

        user = User(id=user_id, email=email, name=name, created_at=datetime.now())
        self._invalidate_cache(user_id)
        logger.info(f"Created user {user_id}: {email}")
        return user

    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user by email and password.

        Args:
            email: User's email address
            password: Plain text password to verify

        Returns:
            User if authentication succeeds, None otherwise
        """
        user_data = self.db.query(
            "SELECT id, email, name, password_hash, created_at, is_active FROM users WHERE email = ?",
            email
        )

        if not user_data or not user_data.get('is_active'):
            logger.warning(f"Authentication failed for {email}: user not found or inactive")
            return None

        if not self._verify_password(password, user_data['password_hash']):
            logger.warning(f"Authentication failed for {email}: invalid password")
            return None

        return User(
            id=user_data['id'],
            email=user_data['email'],
            name=user_data['name'],
            created_at=user_data['created_at'],
            is_active=user_data['is_active']
        )

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with salt."""
        salt = "secure_salt_here"  # In production, use proper salt
        return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return self._hash_password(password) == password_hash

    def _validate_email(self, email: str) -> bool:
        """Basic email validation."""
        return '@' in email and '.' in email.split('@')[1]

    def _invalidate_cache(self, user_id: int) -> None:
        """Clear cached data for a user."""
        if user_id in self._user_cache:
            del self._user_cache[user_id]
        if self.cache:
            self.cache.delete(f"user:{user_id}")
```

---

## Approach 1: Sweep AI / LlamaIndex CodeSplitter (AST + Characters)

**Settings**: `max_chars=1500`

### Algorithm Decision Tree

```
UserService class (2847 chars) > 1500? YES → recurse into children

├── class docstring (147 chars) → bundle
├── __init__ method (428 chars) → bundle with docstring = 575 chars
├── create_user method (1182 chars)
│   └── 575 + 1182 = 1757 > 1500? YES → flush previous, start new
├── authenticate method (892 chars)
│   └── 1182 + 892 = 2074 > 1500? YES → flush, start new
├── _hash_password method (198 chars)
│   └── 892 + 198 = 1090 < 1500? YES → bundle
├── _verify_password method (147 chars)
│   └── 1090 + 147 = 1237 < 1500? YES → bundle
├── _validate_email method (112 chars)
│   └── 1237 + 112 = 1349 < 1500? YES → bundle
└── _invalidate_cache method (189 chars)
    └── 1349 + 189 = 1538 > 1500? YES → flush, add as new
```

### Resulting Chunks

```
┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 1 (575 chars, ~115 tokens)                                │
├─────────────────────────────────────────────────────────────────┤
│ class UserService:                                              │
│     """Service for managing user operations.                    │
│     ...docstring...                                             │
│     """                                                         │
│                                                                 │
│     def __init__(self, db_connection, cache_client=None):       │
│         """Initialize the user service..."""                    │
│         self.db = db_connection                                 │
│         self.cache = cache_client                               │
│         self._user_cache = {}                                   │
│         logger.info("UserService initialized")                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 2 (1182 chars, ~236 tokens)                               │
├─────────────────────────────────────────────────────────────────┤
│     def create_user(self, email: str, name: str, ...) -> User:  │
│         """Create a new user account.                           │
│         ...full docstring...                                    │
│         """                                                     │
│         # Validate email format                                 │
│         if not self._validate_email(email):                     │
│             raise ValueError(...)                               │
│         ...                                                     │
│         return user                                             │
│                                                                 │
│ ✅ COMPLETE FUNCTION - signature + body + docstring together    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 3 (892 chars, ~178 tokens)                                │
├─────────────────────────────────────────────────────────────────┤
│     def authenticate(self, email: str, password: str) -> ...:  │
│         """Authenticate a user by email and password..."""      │
│         user_data = self.db.query(...)                          │
│         ...                                                     │
│         return User(...)                                        │
│                                                                 │
│ ✅ COMPLETE FUNCTION                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 4 (1349 chars, ~270 tokens)                               │
├─────────────────────────────────────────────────────────────────┤
│     def _hash_password(self, password: str) -> str:             │
│         """Hash a password using SHA-256 with salt."""          │
│         salt = "secure_salt_here"                               │
│         return hashlib.sha256(...).hexdigest()                  │
│                                                                 │
│     def _verify_password(self, password: str, ...) -> bool:     │
│         """Verify a password against its hash."""               │
│         return self._hash_password(password) == password_hash   │
│                                                                 │
│     def _validate_email(self, email: str) -> bool:              │
│         """Basic email validation."""                           │
│         return '@' in email and '.' in email.split('@')[1]      │
│                                                                 │
│ ✅ Related helper methods bundled together                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 5 (189 chars, ~38 tokens)                                 │
├─────────────────────────────────────────────────────────────────┤
│     def _invalidate_cache(self, user_id: int) -> None:          │
│         """Clear cached data for a user."""                     │
│         if user_id in self._user_cache:                         │
│             del self._user_cache[user_id]                       │
│         if self.cache:                                          │
│             self.cache.delete(f"user:{user_id}")                │
└─────────────────────────────────────────────────────────────────┘
```

**Result**: 5 chunks, all semantically complete ✅

---

## Approach 2: Token-Counted AST (Hypothetical - Not Widely Implemented)

**Settings**: `max_tokens=300` (using tiktoken cl100k_base)

### Key Difference from Approach 1

Instead of `chars > 1500`, we check `tokens > 300`.

```python
import tiktoken
enc = tiktoken.encoding_for_model("text-embedding-3-small")

def count_tokens(text):
    return len(enc.encode(text))
```

### Algorithm Execution

```
UserService class (~570 tokens) > 300? YES → recurse

├── class docstring (~30 tokens) → bundle
├── __init__ method (~86 tokens) → bundle = 116 tokens
├── create_user method (~236 tokens)
│   └── 116 + 236 = 352 > 300? YES → flush previous, start new
├── authenticate method (~178 tokens)
│   └── 236 + 178 = 414 > 300? YES → flush, start new
├── _hash_password method (~40 tokens)
│   └── 178 + 40 = 218 < 300? YES → bundle
├── _verify_password method (~30 tokens)
│   └── 218 + 30 = 248 < 300? YES → bundle
├── _validate_email method (~22 tokens)
│   └── 248 + 22 = 270 < 300? YES → bundle
└── _invalidate_cache method (~38 tokens)
    └── 270 + 38 = 308 > 300? YES → flush, add as new
```

### Resulting Chunks

```
┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 1 (~116 tokens)                                           │
│ class UserService + __init__                                    │
│ Same as Approach 1, Chunk 1                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 2 (~236 tokens)                                           │
│ create_user method (complete)                                   │
│ Same as Approach 1, Chunk 2                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 3 (~178 tokens)                                           │
│ authenticate method (complete)                                  │
│ Same as Approach 1, Chunk 3                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 4 (~270 tokens)                                           │
│ _hash_password + _verify_password + _validate_email             │
│ Same as Approach 1, Chunk 4                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 5 (~38 tokens)                                            │
│ _invalidate_cache method                                        │
│ Same as Approach 1, Chunk 5                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Result**: Identical to Approach 1!

This is why **token counting doesn't significantly change results for code** - the char-to-token ratio is stable (~5:1).

---

## Approach 3: LangChain RecursiveCharacterTextSplitter

**Settings**: `chunk_size=1500, chunk_overlap=200`

### Algorithm (No AST Awareness)

LangChain uses a **priority list of separators**:

```python
# For Python:
separators = [
    "\nclass ",      # Try to split at class definitions
    "\ndef ",        # Then function definitions
    "\n\tdef ",      # Indented functions (methods)
    "\n\n",          # Double newlines (paragraphs)
    "\n",            # Single newlines
    " ",             # Spaces
    ""               # Characters (last resort)
]
```

### Execution

```
Full text (2847 chars) > 1500? YES

Try split by "\nclass " → only 1 part (the whole class)
Try split by "\ndef " → No matches (methods use "\n    def ")
Try split by "\n\tdef " → No matches (uses spaces, not tabs)

Fall through to "\n\n" (blank lines)...
Split at blank lines between methods...

But wait - there are no blank lines BETWEEN methods in this code!
Each method is separated by just \n, not \n\n.

Fall through to "\n" → Split at every newline...
Combine lines until chunk_size reached...
```

### Resulting Chunks (Approximate)

```
┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 1 (~1450 chars)                                           │
├─────────────────────────────────────────────────────────────────┤
│ # file: user_service.py                                         │
│ from dataclasses import dataclass                               │
│ ...imports...                                                   │
│ @dataclass                                                      │
│ class User:                                                     │
│     ...                                                         │
│ class UserService:                                              │
│     """Service for managing..."""                               │
│     def __init__(self, db_connection, cache_client=None):       │
│         ...                                                     │
│     def create_user(self, email: str, name: str, ...) -> User:  │
│         """Create a new user account.                           │
│         ...docstring...                                         │
│         """                                                     │
│         # Validate email format                                 │
│         if not self._validate_email(email):                     │
│             raise ValueError(...)                               │
│         # Check for existing user                               │
│                                                                 │
│ ⚠️  CUT MID-FUNCTION - create_user continues in next chunk      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 2 (~1480 chars, with 200 char overlap)                    │
├─────────────────────────────────────────────────────────────────┤
│ ...overlap from chunk 1...                                      │
│         # Check for existing user                               │  ← overlap
│         existing = self.db.query(...)                           │
│         if existing:                                            │
│             raise ValueError(...)                               │
│         # Hash password and create user                         │
│         password_hash = self._hash_password(password)           │
│         user_id = self.db.execute(...)                          │
│         user = User(...)                                        │
│         self._invalidate_cache(user_id)                         │
│         logger.info(...)                                        │
│         return user                                             │
│                                                                 │
│     def authenticate(self, email: str, password: str) -> ...:   │
│         """Authenticate a user..."""                            │
│         user_data = self.db.query(...)                          │
│         if not user_data or not user_data.get('is_active'):     │
│             logger.warning(...)                                 │
│             return None                                         │
│         if not self._verify_password(...):                      │
│                                                                 │
│ ⚠️  CUT MID-FUNCTION AGAIN - authenticate continues in chunk 3  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 3 (~890 chars, with 200 char overlap)                     │
├─────────────────────────────────────────────────────────────────┤
│ ...overlap from chunk 2...                                      │
│         if not self._verify_password(...):                      │  ← overlap
│             logger.warning(...)                                 │
│             return None                                         │
│         return User(...)                                        │
│                                                                 │
│     def _hash_password(self, password: str) -> str:             │
│         ...                                                     │
│     def _verify_password(self, ...) -> bool:                    │
│         ...                                                     │
│     def _validate_email(self, email: str) -> bool:              │
│         ...                                                     │
│     def _invalidate_cache(self, user_id: int) -> None:          │
│         ...                                                     │
│                                                                 │
│ ✅ Last chunk happens to contain complete functions             │
└─────────────────────────────────────────────────────────────────┘
```

**Result**: 3 chunks, but 2 of them split functions mid-body! ❌

---

## Side-by-Side Comparison

| Aspect | Sweep AI (AST+Chars) | Token AST | LangChain Recursive |
|--------|---------------------|-----------|---------------------|
| **Chunks** | 5 | 5 | 3 |
| **Complete functions** | 5/5 ✅ | 5/5 ✅ | 3/5 ⚠️ |
| **Split mid-function** | 0 | 0 | 2 |
| **Overlap** | None (not needed) | None | 200 chars |
| **Semantic coherence** | High | High | Medium |
| **Speed** | Fast | Slower (tokenizer) | Fast |
| **Language support** | 113 (tree-sitter) | 113 (tree-sitter) | ~20 (regex) |

---

## What About Our Buggy Implementation?

Our current tree-sitter implementation has a bug where it **recurses too aggressively**:

```
┌─────────────────────────────────────────────────────────────────┐
│ OUR BUG: CHUNK 2 (split incorrectly)                            │
├─────────────────────────────────────────────────────────────────┤
│     def create_user(self, email: str, name: str, ...) -> User:  │
│                                                                 │
│ ❌ ONLY SIGNATURE - 66 chars                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ OUR BUG: CHUNK 3 (body without signature)                       │
├─────────────────────────────────────────────────────────────────┤
│     {                                                           │
│         """Create a new user account..."""                      │
│         # Validate email format                                 │
│         if not self._validate_email(email):                     │
│         ...                                                     │
│         return user                                             │
│                                                                 │
│ ❌ BODY WITHOUT CONTEXT - 1116 chars                            │
└─────────────────────────────────────────────────────────────────┘
```

**Problem**: When the algorithm sees `function_item` (1182 chars < 1500), it should keep it whole. Instead, we're recursing into its children and splitting signature from body.

---

## Conclusion

| Approach | When to Use |
|----------|-------------|
| **AST + Characters (Sweep)** | Best default for code. Fast, semantic boundaries preserved. |
| **AST + Tokens** | When you need exact token control (multi-language, non-code mixed in) |
| **LangChain Recursive** | Quick & dirty, acceptable for well-formatted code with blank lines |

**The key insight**: It's not about tokens vs characters. It's about **respecting semantic boundaries**. AST-based approaches win because they understand code structure.
