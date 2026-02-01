"""
Data Encryption and Security Management

Provides encryption capabilities for sensitive data storage and transmission
in enterprise RAG systems with key management and rotation.
"""
import os
import base64
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import secrets
import json


@dataclass
class EncryptionKey:
    """Represents an encryption key with metadata"""
    key_id: str
    key_data: bytes
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    purpose: str = "general"
    is_active: bool = True


class DataEncryptor:
    """Handles data encryption and decryption operations"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize master key
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = self._generate_master_key()
        
        # Initialize Fernet cipher
        self.fernet = self._create_fernet_cipher()
        
        # Key derivation function
        self.kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # In production, use unique salts
            iterations=100000,
        )
    
    def _generate_master_key(self) -> bytes:
        """Generate a new master encryption key"""
        return Fernet.generate_key()
    
    def _create_fernet_cipher(self) -> Fernet:
        """Create Fernet cipher for symmetric encryption"""
        return Fernet(self.master_key)
    
    def encrypt_text(self, plaintext: str) -> str:
        """Encrypt text data"""
        try:
            encrypted_data = self.fernet.encrypt(plaintext.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """Decrypt text data"""
        try:
            encrypted_data = base64.b64decode(encrypted_text.encode())
            decrypted_data = self.fernet.decrypt(encrypted_data)
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data"""
        json_str = json.dumps(data)
        return self.encrypt_text(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data"""
        json_str = self.decrypt_text(encrypted_data)
        return json.loads(json_str)
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt file contents"""
        output_path = output_path or f"{file_path}.encrypted"
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            encrypted_data = self.fernet.encrypt(file_data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            self.logger.info(f"File encrypted: {file_path} -> {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"File encryption failed: {e}")
            raise
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt file contents"""
        if not output_path:
            output_path = encrypted_file_path.replace('.encrypted', '')
        
        try:
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            self.logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"File decryption failed: {e}")
            raise
    
    def create_secure_hash(self, data: str, salt: Optional[str] = None) -> str:
        """Create secure hash of data"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{data}{salt}".encode()
        digest = hashes.Hash(hashes.SHA256())
        digest.update(combined)
        hash_bytes = digest.finalize()
        
        return f"{salt}:{base64.b64encode(hash_bytes).decode()}"
    
    def verify_hash(self, data: str, stored_hash: str) -> bool:
        """Verify data against stored hash"""
        try:
            salt, hash_b64 = stored_hash.split(':', 1)
            expected_hash = self.create_secure_hash(data, salt)
            return expected_hash == stored_hash
        except Exception:
            return False


class EncryptionManager:
    """Manages encryption keys and provides enterprise encryption services"""
    
    def __init__(self, key_storage_path: str = "./encryption_keys"):
        self.key_storage_path = key_storage_path
        self.logger = logging.getLogger(__name__)
        
        # Ensure key storage directory exists
        os.makedirs(key_storage_path, exist_ok=True)
        
        # Key registry
        self.keys: Dict[str, EncryptionKey] = {}
        
        # Load existing keys
        self._load_keys()
        
        # Create default encryption key if none exist
        if not self.keys:
            self._create_default_key()
    
    def _load_keys(self):
        """Load encryption keys from storage"""
        key_registry_path = os.path.join(self.key_storage_path, "key_registry.json")
        
        if os.path.exists(key_registry_path):
            try:
                with open(key_registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                for key_id, key_info in registry_data.items():
                    key_file_path = os.path.join(self.key_storage_path, f"{key_id}.key")
                    if os.path.exists(key_file_path):
                        with open(key_file_path, 'rb') as f:
                            key_data = f.read()
                        
                        encryption_key = EncryptionKey(
                            key_id=key_id,
                            key_data=key_data,
                            algorithm=key_info.get('algorithm', 'fernet'),
                            created_at=datetime.fromisoformat(key_info['created_at']),
                            expires_at=datetime.fromisoformat(key_info['expires_at']) if key_info.get('expires_at') else None,
                            purpose=key_info.get('purpose', 'general'),
                            is_active=key_info.get('is_active', True)
                        )
                        
                        self.keys[key_id] = encryption_key
                
                self.logger.info(f"Loaded {len(self.keys)} encryption keys")
            except Exception as e:
                self.logger.error(f"Failed to load encryption keys: {e}")
    
    def _save_keys(self):
        """Save encryption keys to storage"""
        key_registry_path = os.path.join(self.key_storage_path, "key_registry.json")
        
        registry_data = {}
        for key_id, key in self.keys.items():
            # Save key data to separate file
            key_file_path = os.path.join(self.key_storage_path, f"{key_id}.key")
            with open(key_file_path, 'wb') as f:
                f.write(key.key_data)
            
            # Save metadata to registry
            registry_data[key_id] = {
                'algorithm': key.algorithm,
                'created_at': key.created_at.isoformat(),
                'expires_at': key.expires_at.isoformat() if key.expires_at else None,
                'purpose': key.purpose,
                'is_active': key.is_active
            }
        
        with open(key_registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        self.logger.info(f"Saved {len(self.keys)} encryption keys")
    
    def _create_default_key(self):
        """Create default encryption key"""
        key_id = "default"
        key_data = Fernet.generate_key()
        
        encryption_key = EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            algorithm="fernet",
            created_at=datetime.now(),
            purpose="general"
        )
        
        self.keys[key_id] = encryption_key
        self._save_keys()
        
        self.logger.info("Created default encryption key")
    
    def create_key(self,
                   key_id: str,
                   purpose: str = "general",
                   algorithm: str = "fernet",
                   expires_in_days: Optional[int] = None) -> EncryptionKey:
        """Create a new encryption key"""
        if key_id in self.keys:
            raise ValueError(f"Key {key_id} already exists")
        
        # Generate key based on algorithm
        if algorithm == "fernet":
            key_data = Fernet.generate_key()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        encryption_key = EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            algorithm=algorithm,
            created_at=datetime.now(),
            expires_at=expires_at,
            purpose=purpose
        )
        
        self.keys[key_id] = encryption_key
        self._save_keys()
        
        self.logger.info(f"Created encryption key: {key_id} for purpose: {purpose}")
        return encryption_key
    
    def get_encryptor(self, key_id: str = "default") -> DataEncryptor:
        """Get encryptor for specific key"""
        if key_id not in self.keys:
            raise ValueError(f"Key {key_id} not found")
        
        key = self.keys[key_id]
        if not key.is_active:
            raise ValueError(f"Key {key_id} is not active")
        
        if key.expires_at and datetime.now() > key.expires_at:
            raise ValueError(f"Key {key_id} has expired")
        
        return DataEncryptor(key.key_data.decode())
    
    def rotate_key(self, key_id: str) -> EncryptionKey:
        """Rotate an existing encryption key"""
        if key_id not in self.keys:
            raise ValueError(f"Key {key_id} not found")
        
        old_key = self.keys[key_id]
        
        # Create new key with same properties
        new_key = self.create_key(
            key_id=f"{key_id}_new",
            purpose=old_key.purpose,
            algorithm=old_key.algorithm,
            expires_in_days=365  # Default 1 year expiration
        )
        
        # Mark old key as inactive
        old_key.is_active = False
        
        # Replace with new key
        self.keys[key_id] = new_key
        self.keys[f"{key_id}_old"] = old_key
        
        self._save_keys()
        
        self.logger.info(f"Rotated encryption key: {key_id}")
        return new_key
    
    def deactivate_key(self, key_id: str):
        """Deactivate an encryption key"""
        if key_id in self.keys:
            self.keys[key_id].is_active = False
            self._save_keys()
            self.logger.info(f"Deactivated encryption key: {key_id}")
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all encryption keys"""
        keys_info = []
        for key_id, key in self.keys.items():
            keys_info.append({
                'key_id': key_id,
                'algorithm': key.algorithm,
                'purpose': key.purpose,
                'created_at': key.created_at.isoformat(),
                'expires_at': key.expires_at.isoformat() if key.expires_at else None,
                'is_active': key.is_active,
                'is_expired': key.expires_at and datetime.now() > key.expires_at if key.expires_at else False
            })
        
        return keys_info
    
    def cleanup_expired_keys(self):
        """Remove expired keys from storage"""
        expired_keys = []
        
        for key_id, key in self.keys.items():
            if key.expires_at and datetime.now() > key.expires_at:
                expired_keys.append(key_id)
        
        for key_id in expired_keys:
            # Remove key file
            key_file_path = os.path.join(self.key_storage_path, f"{key_id}.key")
            if os.path.exists(key_file_path):
                os.remove(key_file_path)
            
            # Remove from registry
            del self.keys[key_id]
        
        if expired_keys:
            self._save_keys()
            self.logger.info(f"Cleaned up {len(expired_keys)} expired keys")
    
    def export_key_for_backup(self, key_id: str, password: str) -> str:
        """Export key for secure backup"""
        if key_id not in self.keys:
            raise ValueError(f"Key {key_id} not found")
        
        key = self.keys[key_id]
        
        # Encrypt key data with password
        encryptor = DataEncryptor(password)
        encrypted_key_data = encryptor.encrypt_text(key.key_data.decode())
        
        backup_data = {
            'key_id': key_id,
            'algorithm': key.algorithm,
            'purpose': key.purpose,
            'created_at': key.created_at.isoformat(),
            'encrypted_key_data': encrypted_key_data
        }
        
        return base64.b64encode(json.dumps(backup_data).encode()).decode()
    
    def import_key_from_backup(self, backup_data: str, password: str) -> EncryptionKey:
        """Import key from secure backup"""
        try:
            # Decode backup data
            backup_json = base64.b64decode(backup_data.encode()).decode()
            backup_dict = json.loads(backup_json)
            
            # Decrypt key data
            encryptor = DataEncryptor(password)
            key_data = encryptor.decrypt_text(backup_dict['encrypted_key_data']).encode()
            
            # Create encryption key
            encryption_key = EncryptionKey(
                key_id=backup_dict['key_id'],
                key_data=key_data,
                algorithm=backup_dict['algorithm'],
                created_at=datetime.fromisoformat(backup_dict['created_at']),
                purpose=backup_dict['purpose']
            )
            
            # Store key
            self.keys[encryption_key.key_id] = encryption_key
            self._save_keys()
            
            self.logger.info(f"Imported encryption key: {encryption_key.key_id}")
            return encryption_key
            
        except Exception as e:
            self.logger.error(f"Failed to import key from backup: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Test encryption manager
    manager = EncryptionManager()
    
    # Create test encryptor
    encryptor = manager.get_encryptor()
    
    # Test text encryption
    test_text = "This is sensitive enterprise data that needs protection."
    encrypted = encryptor.encrypt_text(test_text)
    decrypted = encryptor.decrypt_text(encrypted)
    
    print(f"Original: {test_text}")
    print(f"Encrypted: {encrypted[:50]}...")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {test_text == decrypted}")
    
    # Test dictionary encryption
    test_dict = {
        "user_id": "user123",
        "personal_info": {
            "ssn": "123-45-6789",
            "email": "user@company.com"
        }
    }
    
    encrypted_dict = encryptor.encrypt_dict(test_dict)
    decrypted_dict = encryptor.decrypt_dict(encrypted_dict)
    
    print(f"Dictionary encrypted and decrypted successfully: {test_dict == decrypted_dict}")
    
    # List all keys
    keys = manager.list_keys()
    print(f"Available keys: {[k['key_id'] for k in keys]}")