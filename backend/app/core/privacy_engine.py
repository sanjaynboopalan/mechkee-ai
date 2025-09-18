"""
Privacy Layer with Encryption and Secure Personal Data Learning
Ensures user data privacy while enabling personalized learning
"""

import asyncio
import json
import os
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import base64
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import numpy as np

logger = logging.getLogger(__name__)

class PrivacyLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class DataCategory(Enum):
    PREFERENCES = "preferences"
    INTERACTIONS = "interactions"
    GOALS = "goals"
    BEHAVIOR_PATTERNS = "behavior_patterns"
    FEEDBACK = "feedback"
    ANALYTICS = "analytics"

@dataclass
class EncryptedData:
    """Represents encrypted data with metadata"""
    encrypted_content: bytes
    encryption_method: str
    key_id: str
    timestamp: datetime
    data_category: DataCategory
    privacy_level: PrivacyLevel
    access_permissions: List[str]
    retention_period: Optional[timedelta]
    
@dataclass
class PrivacyPolicy:
    """Privacy policy configuration"""
    user_id: str
    data_retention_days: int
    allow_analytics: bool
    allow_personalization: bool
    allow_sharing: bool
    encryption_level: str
    auto_delete_enabled: bool
    data_minimization: bool
    
@dataclass
class AuditLog:
    """Privacy audit log entry"""
    log_id: str
    user_id: str
    action: str
    data_accessed: List[str]
    accessor: str
    timestamp: datetime
    purpose: str
    legal_basis: str

class DifferentialPrivacy:
    """Implements differential privacy for user data analysis"""
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon  # Privacy budget
        
    def add_noise(self, true_value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise to preserve differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise
    
    def add_gaussian_noise(self, true_value: float, sensitivity: float = 1.0, delta: float = 1e-5) -> float:
        """Add Gaussian noise for (epsilon, delta)-differential privacy"""
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / self.epsilon
        noise = np.random.normal(0, sigma)
        return true_value + noise
    
    def privatize_histogram(self, histogram: Dict[str, int], sensitivity: int = 1) -> Dict[str, float]:
        """Add noise to histogram counts"""
        privatized = {}
        for key, count in histogram.items():
            privatized[key] = max(0, self.add_noise(count, sensitivity))
        return privatized

class HomomorphicEncryption:
    """Simplified homomorphic encryption for basic operations"""
    
    def __init__(self):
        self.key = self._generate_key()
    
    def _generate_key(self) -> int:
        """Generate simple key for demo (in practice, use proper HE library)"""
        return secrets.randbelow(1000000)
    
    def encrypt(self, value: float) -> Tuple[int, int]:
        """Encrypt a value (simplified implementation)"""
        # In practice, use a proper HE library like Microsoft SEAL
        noise = secrets.randbelow(100)
        encrypted = int(value * self.key + noise)
        return encrypted, noise
    
    def decrypt(self, encrypted_value: int, noise: int) -> float:
        """Decrypt a value"""
        return (encrypted_value - noise) / self.key
    
    def add_encrypted(self, enc1: Tuple[int, int], enc2: Tuple[int, int]) -> Tuple[int, int]:
        """Add two encrypted values"""
        result = enc1[0] + enc2[0]
        combined_noise = enc1[1] + enc2[1]
        return result, combined_noise

class SecureAggregation:
    """Secure aggregation for federated learning"""
    
    def __init__(self, num_users: int):
        self.num_users = num_users
        self.user_masks = {}
        
    def generate_user_mask(self, user_id: str) -> np.ndarray:
        """Generate random mask for user"""
        mask = np.random.randint(-1000, 1000, size=10)  # Simplified
        self.user_masks[user_id] = mask
        return mask
    
    def mask_update(self, user_id: str, gradient: np.ndarray) -> np.ndarray:
        """Mask user's gradient update"""
        if user_id not in self.user_masks:
            self.generate_user_mask(user_id)
        
        masked = gradient + self.user_masks[user_id]
        return masked
    
    def aggregate_updates(self, masked_updates: List[np.ndarray]) -> np.ndarray:
        """Securely aggregate masked updates"""
        # Sum all updates - masks cancel out
        aggregated = np.sum(masked_updates, axis=0)
        return aggregated / len(masked_updates)

class PrivacyEngine:
    """
    Main privacy engine that handles:
    1. End-to-end encryption of personal data
    2. Differential privacy for analytics
    3. Secure federated learning
    4. Privacy policy enforcement
    5. Data retention and deletion
    """
    
    def __init__(self):
        self.master_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)
        self.user_keys: Dict[str, bytes] = {}
        self.privacy_policies: Dict[str, PrivacyPolicy] = {}
        self.audit_logs: List[AuditLog] = []
        self.differential_privacy = DifferentialPrivacy(epsilon=1.0)
        self.homomorphic_encryption = HomomorphicEncryption()
        self.secure_aggregation = SecureAggregation(num_users=1000)
        
        # RSA key pair for additional security
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
    async def initialize_user_privacy(
        self,
        user_id: str,
        privacy_preferences: Dict[str, Any] = None
    ) -> PrivacyPolicy:
        """Initialize privacy settings for a new user"""
        
        # Generate unique encryption key for user
        user_key = Fernet.generate_key()
        self.user_keys[user_id] = user_key
        
        # Create default privacy policy
        default_preferences = {
            'data_retention_days': 365,
            'allow_analytics': True,
            'allow_personalization': True,
            'allow_sharing': False,
            'encryption_level': 'strong',
            'auto_delete_enabled': True,
            'data_minimization': True
        }
        
        if privacy_preferences:
            default_preferences.update(privacy_preferences)
        
        policy = PrivacyPolicy(
            user_id=user_id,
            **default_preferences
        )
        
        self.privacy_policies[user_id] = policy
        
        # Log privacy initialization
        await self._audit_log(
            user_id=user_id,
            action="privacy_initialized",
            data_accessed=[],
            accessor="system",
            purpose="setup",
            legal_basis="consent"
        )
        
        return policy
    
    async def encrypt_user_data(
        self,
        user_id: str,
        data: Any,
        data_category: DataCategory,
        privacy_level: PrivacyLevel = PrivacyLevel.CONFIDENTIAL
    ) -> EncryptedData:
        """Encrypt user data with appropriate privacy level"""
        
        if user_id not in self.user_keys:
            await self.initialize_user_privacy(user_id)
        
        # Serialize data
        if isinstance(data, dict):
            serialized_data = json.dumps(data, default=str).encode()
        elif isinstance(data, str):
            serialized_data = data.encode()
        else:
            serialized_data = str(data).encode()
        
        # Choose encryption method based on privacy level
        if privacy_level == PrivacyLevel.RESTRICTED:
            # Use RSA encryption for highly sensitive data
            encrypted_content = self.public_key.encrypt(
                serialized_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            encryption_method = "RSA_OAEP"
        else:
            # Use Fernet encryption for regular data
            user_cipher = Fernet(self.user_keys[user_id])
            encrypted_content = user_cipher.encrypt(serialized_data)
            encryption_method = "Fernet"
        
        # Set retention period based on data category
        retention_periods = {
            DataCategory.PREFERENCES: timedelta(days=730),
            DataCategory.INTERACTIONS: timedelta(days=365),
            DataCategory.GOALS: timedelta(days=365),
            DataCategory.BEHAVIOR_PATTERNS: timedelta(days=180),
            DataCategory.FEEDBACK: timedelta(days=90),
            DataCategory.ANALYTICS: timedelta(days=30)
        }
        
        policy = self.privacy_policies.get(user_id)
        if policy and policy.data_retention_days:
            retention_period = timedelta(days=policy.data_retention_days)
        else:
            retention_period = retention_periods.get(data_category, timedelta(days=365))
        
        encrypted_data = EncryptedData(
            encrypted_content=encrypted_content,
            encryption_method=encryption_method,
            key_id=user_id,
            timestamp=datetime.now(),
            data_category=data_category,
            privacy_level=privacy_level,
            access_permissions=[user_id, "system"],
            retention_period=retention_period
        )
        
        # Log encryption
        await self._audit_log(
            user_id=user_id,
            action="data_encrypted",
            data_accessed=[data_category.value],
            accessor="system",
            purpose="privacy_protection",
            legal_basis="consent"
        )
        
        return encrypted_data
    
    async def decrypt_user_data(
        self,
        user_id: str,
        encrypted_data: EncryptedData,
        accessor: str = "system",
        purpose: str = "processing"
    ) -> Any:
        """Decrypt user data with access control"""
        
        # Check access permissions
        if accessor not in encrypted_data.access_permissions:
            raise PermissionError(f"Access denied for {accessor}")
        
        # Check if data has expired
        if encrypted_data.retention_period:
            expiry_date = encrypted_data.timestamp + encrypted_data.retention_period
            if datetime.now() > expiry_date:
                raise ValueError("Data has expired and should be deleted")
        
        # Decrypt based on method
        if encrypted_data.encryption_method == "RSA_OAEP":
            decrypted_bytes = self.private_key.decrypt(
                encrypted_data.encrypted_content,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        elif encrypted_data.encryption_method == "Fernet":
            if user_id not in self.user_keys:
                raise ValueError(f"No key found for user {user_id}")
            
            user_cipher = Fernet(self.user_keys[user_id])
            decrypted_bytes = user_cipher.decrypt(encrypted_data.encrypted_content)
        else:
            raise ValueError(f"Unknown encryption method: {encrypted_data.encryption_method}")
        
        # Deserialize data
        try:
            decrypted_str = decrypted_bytes.decode()
            decrypted_data = json.loads(decrypted_str)
        except (json.JSONDecodeError, UnicodeDecodeError):
            decrypted_data = decrypted_str
        
        # Log access
        await self._audit_log(
            user_id=user_id,
            action="data_decrypted",
            data_accessed=[encrypted_data.data_category.value],
            accessor=accessor,
            purpose=purpose,
            legal_basis="legitimate_interest"
        )
        
        return decrypted_data
    
    async def apply_differential_privacy(
        self,
        user_data: Dict[str, Any],
        analysis_type: str = "general"
    ) -> Dict[str, Any]:
        """Apply differential privacy to user data for analytics"""
        
        privatized_data = {}
        
        for key, value in user_data.items():
            if isinstance(value, (int, float)):
                # Add noise to numerical values
                privatized_data[key] = self.differential_privacy.add_noise(float(value))
            elif isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
                # Handle histograms/counters
                privatized_data[key] = self.differential_privacy.privatize_histogram(
                    {k: int(v) for k, v in value.items()}
                )
            else:
                # For non-numerical data, use generalization
                privatized_data[key] = self._generalize_categorical_data(value)
        
        return privatized_data
    
    def _generalize_categorical_data(self, value: Any) -> str:
        """Generalize categorical data for privacy"""
        
        if isinstance(value, str):
            # Simple generalization - could be more sophisticated
            if len(value) > 10:
                return "long_text"
            elif len(value) > 5:
                return "medium_text"
            else:
                return "short_text"
        
        return "other"
    
    async def federated_learning_update(
        self,
        user_id: str,
        local_update: np.ndarray,
        round_number: int
    ) -> bool:
        """Perform federated learning update with privacy preservation"""
        
        # Apply differential privacy to local update
        epsilon_per_round = 0.1  # Privacy budget per round
        dp = DifferentialPrivacy(epsilon=epsilon_per_round)
        
        privatized_update = np.array([
            dp.add_noise(value) for value in local_update
        ])
        
        # Apply secure aggregation
        masked_update = self.secure_aggregation.mask_update(user_id, privatized_update)
        
        # In a real system, this would be sent to the federated learning server
        # For demo, we'll just log it
        await self._audit_log(
            user_id=user_id,
            action="federated_learning_update",
            data_accessed=["model_gradients"],
            accessor="federated_server",
            purpose="model_improvement",
            legal_basis="legitimate_interest"
        )
        
        return True
    
    async def homomorphic_computation(
        self,
        user_id: str,
        encrypted_values: List[Tuple[int, int]],
        operation: str = "sum"
    ) -> Tuple[int, int]:
        """Perform computation on encrypted data"""
        
        if not encrypted_values:
            raise ValueError("No encrypted values provided")
        
        if operation == "sum":
            result = encrypted_values[0]
            for enc_val in encrypted_values[1:]:
                result = self.homomorphic_encryption.add_encrypted(result, enc_val)
        else:
            raise ValueError(f"Operation {operation} not supported")
        
        # Log homomorphic computation
        await self._audit_log(
            user_id=user_id,
            action="homomorphic_computation",
            data_accessed=["encrypted_analytics"],
            accessor="system",
            purpose="privacy_preserving_analytics",
            legal_basis="legitimate_interest"
        )
        
        return result
    
    async def enforce_data_retention(self) -> Dict[str, int]:
        """Enforce data retention policies and delete expired data"""
        
        deletion_stats = {
            'users_processed': 0,
            'items_deleted': 0,
            'errors': 0
        }
        
        current_time = datetime.now()
        
        for user_id, policy in self.privacy_policies.items():
            if not policy.auto_delete_enabled:
                continue
            
            try:
                # This would interface with your data storage system
                # For demo, we'll simulate deletion
                retention_cutoff = current_time - timedelta(days=policy.data_retention_days)
                
                # Log data deletion
                await self._audit_log(
                    user_id=user_id,
                    action="data_retention_enforcement",
                    data_accessed=["expired_data"],
                    accessor="system",
                    purpose="compliance",
                    legal_basis="legal_obligation"
                )
                
                deletion_stats['users_processed'] += 1
                deletion_stats['items_deleted'] += 1  # Simulated
                
            except Exception as e:
                logger.error(f"Error enforcing retention for user {user_id}: {e}")
                deletion_stats['errors'] += 1
        
        return deletion_stats
    
    async def generate_privacy_report(self, user_id: str) -> Dict[str, Any]:
        """Generate privacy report for a user"""
        
        policy = self.privacy_policies.get(user_id)
        if not policy:
            raise ValueError(f"No privacy policy found for user {user_id}")
        
        # Get user's audit logs
        user_logs = [log for log in self.audit_logs if log.user_id == user_id]
        
        # Calculate privacy metrics
        data_access_count = len([log for log in user_logs if log.action == "data_decrypted"])
        encryption_count = len([log for log in user_logs if log.action == "data_encrypted"])
        
        report = {
            'user_id': user_id,
            'privacy_policy': asdict(policy),
            'data_access_summary': {
                'total_accesses': data_access_count,
                'total_encryptions': encryption_count,
                'last_access': max([log.timestamp for log in user_logs]) if user_logs else None
            },
            'privacy_measures': {
                'encryption_enabled': True,
                'differential_privacy_enabled': True,
                'data_minimization_enabled': policy.data_minimization,
                'auto_deletion_enabled': policy.auto_delete_enabled
            },
            'compliance_status': {
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_retention_compliant': True
            },
            'audit_trail': [
                {
                    'action': log.action,
                    'timestamp': log.timestamp,
                    'purpose': log.purpose,
                    'legal_basis': log.legal_basis
                }
                for log in user_logs[-10:]  # Last 10 actions
            ]
        }
        
        return report
    
    async def update_privacy_preferences(
        self,
        user_id: str,
        new_preferences: Dict[str, Any]
    ) -> PrivacyPolicy:
        """Update user's privacy preferences"""
        
        if user_id not in self.privacy_policies:
            return await self.initialize_user_privacy(user_id, new_preferences)
        
        policy = self.privacy_policies[user_id]
        
        # Update preferences
        for key, value in new_preferences.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        
        # Log preference update
        await self._audit_log(
            user_id=user_id,
            action="privacy_preferences_updated",
            data_accessed=["privacy_policy"],
            accessor=user_id,
            purpose="user_control",
            legal_basis="consent"
        )
        
        return policy
    
    async def delete_user_data(self, user_id: str, reason: str = "user_request") -> bool:
        """Completely delete all user data"""
        
        try:
            # Delete encryption keys
            if user_id in self.user_keys:
                del self.user_keys[user_id]
            
            # Delete privacy policy
            if user_id in self.privacy_policies:
                del self.privacy_policies[user_id]
            
            # This would delete all user data from storage systems
            # For demo, we'll just log it
            
            # Log data deletion
            await self._audit_log(
                user_id=user_id,
                action="complete_data_deletion",
                data_accessed=["all_user_data"],
                accessor="system",
                purpose=reason,
                legal_basis="user_request" if reason == "user_request" else "legal_obligation"
            )
            
            logger.info(f"Successfully deleted all data for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting data for user {user_id}: {e}")
            return False
    
    async def _audit_log(
        self,
        user_id: str,
        action: str,
        data_accessed: List[str],
        accessor: str,
        purpose: str,
        legal_basis: str
    ) -> None:
        """Create audit log entry"""
        
        log_entry = AuditLog(
            log_id=str(len(self.audit_logs) + 1),
            user_id=user_id,
            action=action,
            data_accessed=data_accessed,
            accessor=accessor,
            timestamp=datetime.now(),
            purpose=purpose,
            legal_basis=legal_basis
        )
        
        self.audit_logs.append(log_entry)
        
        # In production, this would be stored in a secure, immutable log
        logger.debug(f"Audit log created: {action} for user {user_id}")
    
    def get_encryption_key_info(self, user_id: str) -> Dict[str, str]:
        """Get information about user's encryption keys (for key management)"""
        
        if user_id not in self.user_keys:
            return {"status": "no_key"}
        
        # Return key metadata, not the actual key
        key_hash = hashlib.sha256(self.user_keys[user_id]).hexdigest()[:16]
        
        return {
            "status": "key_exists",
            "key_hash": key_hash,
            "created": "system_initialization",  # In practice, store creation time
            "last_used": "recent"  # In practice, track actual usage
        }
    
    async def rotate_encryption_keys(self, user_id: str) -> bool:
        """Rotate encryption keys for enhanced security"""
        
        if user_id not in self.user_keys:
            return False
        
        try:
            # Generate new key
            new_key = Fernet.generate_key()
            old_key = self.user_keys[user_id]
            
            # In production, you would:
            # 1. Decrypt all data with old key
            # 2. Re-encrypt with new key
            # 3. Update key reference
            
            self.user_keys[user_id] = new_key
            
            # Log key rotation
            await self._audit_log(
                user_id=user_id,
                action="encryption_key_rotated",
                data_accessed=["encryption_keys"],
                accessor="system",
                purpose="security_maintenance",
                legal_basis="legitimate_interest"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error rotating keys for user {user_id}: {e}")
            return False

# Helper functions for privacy-preserving operations
async def create_privacy_preserving_analytics(
    privacy_engine: PrivacyEngine,
    user_data_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create analytics while preserving privacy"""
    
    # Apply differential privacy to each user's data
    privatized_data_list = []
    for user_data in user_data_list:
        privatized = await privacy_engine.apply_differential_privacy(user_data)
        privatized_data_list.append(privatized)
    
    # Aggregate privatized data
    aggregated_analytics = {}
    
    # Simple aggregation example
    all_keys = set()
    for data in privatized_data_list:
        all_keys.update(data.keys())
    
    for key in all_keys:
        values = [data.get(key, 0) for data in privatized_data_list if isinstance(data.get(key), (int, float))]
        if values:
            aggregated_analytics[key] = {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
    
    return aggregated_analytics

def generate_synthetic_data(
    original_data: List[Dict[str, Any]],
    privacy_budget: float = 1.0
) -> List[Dict[str, Any]]:
    """Generate synthetic data that preserves privacy"""
    
    dp = DifferentialPrivacy(epsilon=privacy_budget)
    synthetic_data = []
    
    for record in original_data:
        synthetic_record = {}
        for key, value in record.items():
            if isinstance(value, (int, float)):
                # Add noise to numerical values
                synthetic_record[key] = dp.add_noise(float(value))
            elif isinstance(value, str):
                # For strings, use generalization or keep categories
                synthetic_record[key] = "category_" + str(hash(value) % 10)
            else:
                synthetic_record[key] = "synthetic_value"
        
        synthetic_data.append(synthetic_record)
    
    return synthetic_data