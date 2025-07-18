use crate::{
    ast_lowering::resolve::{Declaration, Declarations},
    hir,
    ty::{Gcx, Ty},
};
use solar_ast::StateMutability as SM;
use solar_interface::{Span, Symbol, kw, sym};

pub(crate) mod members;
pub use members::{Member, MemberList};

pub(crate) fn scopes() -> (Declarations, Box<[Option<Declarations>; Builtin::COUNT]>) {
    let global = declarations(Builtin::global());
    let members_map = Box::new(std::array::from_fn(|i| {
        Some(declarations(Builtin::from_index(i).unwrap().members()?))
    }));
    (global, members_map)
}

fn declarations(builtins: impl IntoIterator<Item = Builtin>) -> Declarations {
    let mut declarations = Declarations::new();
    for builtin in builtins {
        let decl = Declaration { res: hir::Res::Builtin(builtin), span: Span::DUMMY };
        declarations.declarations.entry(builtin.name()).or_default().push(decl);
    }
    declarations
}

type Primitive = u8;

macro_rules! declare_builtins {
    (|$gcx:ident| $($(#[$variant_attr:meta])* $variant_name:ident => $sym:ident::$name:ident => $ty:expr;)*) => {
        /// A compiler builtin.
        #[repr(u8)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Builtin {
            $(
                $(#[$variant_attr])*
                $variant_name,
            )*
        }

        impl Builtin {
            // #[doc(hidden)]
            // const ALL: [Self; Self::COUNT] = [$(Self::$variant_name,)*];
            pub const COUNT: usize = 0 $(+ { let _ = Builtin::$variant_name; 1 })*;

            /// Returns the symbol of the builtin.
            pub fn name(self) -> Symbol {
                match self {
                    $(
                        Builtin::$variant_name => $sym::$name,
                    )*
                }
            }

            /// Returns the type of the builtin.
            pub fn ty(self, $gcx: Gcx<'_>) -> Ty<'_> {
                match self {
                    $(
                        Builtin::$variant_name => $ty,
                    )*
                }
            }
        }
    };
}

// https://docs.soliditylang.org/en/latest/units-and-global-variables.html
// https://github.com/ethereum/solidity/blob/b136829e4998a9f0ebc6ca87b7ba45362fe83ba0/libsolidity/analysis/GlobalContext.cpp#L73
// NOTE: Order matters, see functions below.
declare_builtins! {
    |gcx|

    // Global
    Blockhash              => kw::Blockhash
                           => gcx.mk_builtin_fn(&[gcx.types.uint(256)], SM::View, &[gcx.types.fixed_bytes(32)]);
    Blobhash               => kw::Blobhash
                           => gcx.mk_builtin_fn(&[gcx.types.uint(256)], SM::View, &[gcx.types.fixed_bytes(32)]);

    Gasleft                => sym::gasleft
                           => gcx.mk_builtin_fn(&[], SM::View, &[gcx.types.uint(256)]);
    Selfdestruct           => kw::Selfdestruct
                           => gcx.mk_builtin_fn(&[gcx.types.address_payable], SM::NonPayable, &[]);

    Assert                 => sym::assert
                           => gcx.mk_builtin_fn(&[gcx.types.bool], SM::Pure, &[]);
    Require                => sym::require
                           => gcx.mk_builtin_fn(&[gcx.types.bool], SM::Pure, &[]);
    RequireMsg             => sym::require
                           => gcx.mk_builtin_fn(&[gcx.types.bool, gcx.types.string_ref.memory], SM::Pure, &[]);
    // RequireErr             => sym::require
    //                        => gcx.mk_builtin_fn(&[gcx.types.bool, gcx.type_of()], SM::Pure, &[]);
    Revert                 => kw::Revert
                           => gcx.mk_builtin_fn(&[], SM::Pure, &[]);
    RevertMsg              => kw::Revert
                           => gcx.mk_builtin_fn(&[gcx.types.string], SM::Pure, &[]);

    AddMod                 => kw::Addmod
                           => gcx.mk_builtin_fn(&[gcx.types.uint(256), gcx.types.uint(256), gcx.types.uint(256)], SM::Pure, &[gcx.types.uint(256)]);
    MulMod                 => kw::Mulmod
                           => gcx.mk_builtin_fn(&[gcx.types.uint(256), gcx.types.uint(256), gcx.types.uint(256)], SM::Pure, &[gcx.types.uint(256)]);

    Keccak256              => kw::Keccak256
                           => gcx.mk_builtin_fn(&[gcx.types.bytes_ref.memory], SM::Pure, &[gcx.types.fixed_bytes(32)]);
    Sha256                 => sym::sha256
                           => gcx.mk_builtin_fn(&[gcx.types.bytes_ref.memory], SM::Pure, &[gcx.types.fixed_bytes(32)]);
    Ripemd160              => sym::ripemd160
                           => gcx.mk_builtin_fn(&[gcx.types.bytes_ref.memory], SM::Pure, &[gcx.types.fixed_bytes(20)]);
    EcRecover              => sym::ecrecover
                           => gcx.mk_builtin_fn(&[gcx.types.fixed_bytes(32), gcx.types.uint(8), gcx.types.fixed_bytes(32), gcx.types.fixed_bytes(32)], SM::Pure, &[gcx.types.address]);

    Block                  => sym::block
                           => gcx.mk_builtin_mod(Self::Block);
    Msg                    => sym::msg
                           => gcx.mk_builtin_mod(Self::Msg);
    Tx                     => sym::tx
                           => gcx.mk_builtin_mod(Self::Tx);
    Abi                    => sym::abi
                           => gcx.mk_builtin_mod(Self::Abi);

    // Contract
    This                   => sym::this   => unreachable!();
    Super                  => sym::super_ => unreachable!();

    // `block`
    BlockCoinbase          => kw::Coinbase
                           => gcx.types.address_payable;
    BlockTimestamp         => kw::Timestamp
                           => gcx.types.uint(256);
    BlockDifficulty        => kw::Difficulty
                           => gcx.types.uint(256);
    BlockPrevrandao        => kw::Prevrandao
                           => gcx.types.uint(256);
    BlockNumber            => kw::Number
                           => gcx.types.uint(256);
    BlockGaslimit          => kw::Gaslimit
                           => gcx.types.uint(256);
    BlockChainid           => kw::Chainid
                           => gcx.types.uint(256);
    BlockBasefee           => kw::Basefee
                           => gcx.types.uint(256);
    BlockBlobbasefee       => kw::Blobbasefee
                           => gcx.types.uint(256);

    // `msg`
    MsgSender              => sym::sender
                           => gcx.types.address;
    MsgGas                 => kw::Gas
                           => gcx.types.uint(256);
    MsgValue               => sym::value
                           => gcx.types.uint(256);
    MsgData                => sym::data
                           => gcx.types.bytes_ref.calldata;
    MsgSig                 => sym::sig
                           => gcx.types.fixed_bytes(4);

    // `tx`
    TxOrigin               => kw::Origin
                           => gcx.types.address;
    TxGasPrice             => kw::Gasprice
                           => gcx.types.uint(256);

    // `abi`
    // TODO                => `(T...) pure returns(bytes memory)`
    AbiEncode              => sym::encode
                           => gcx.mk_builtin_fn(&[], SM::Pure, &[gcx.types.bytes_ref.memory]);
    // TODO                => `(T...) pure returns(bytes memory)`
    AbiEncodePacked        => sym::encodePacked
                           => gcx.mk_builtin_fn(&[], SM::Pure, &[gcx.types.bytes_ref.memory]);
    // TODO                => `(bytes4, T...) pure returns(bytes memory)`
    AbiEncodeWithSelector  => sym::encodeWithSelector
                           => gcx.mk_builtin_fn(&[], SM::Pure, &[gcx.types.bytes_ref.memory]);
    // TODO                => `(F, T...) pure returns(bytes memory)`
    AbiEncodeCall          => sym::encodeCall
                           => gcx.mk_builtin_fn(&[], SM::Pure, &[gcx.types.bytes_ref.memory]);
    // TODO                => `(string memory, T...) pure returns(bytes memory)`
    AbiEncodeWithSignature => sym::encodeWithSignature
                           => gcx.mk_builtin_fn(&[], SM::Pure, &[gcx.types.bytes_ref.memory]);
    // TODO                => `(bytes memory, (T...)) pure returns(T...)`
    AbiDecode              => sym::decode
                           => gcx.mk_builtin_fn(&[], SM::Pure, &[]);

    // --- impls ---

    AddressBalance         => kw::Balance
                           => gcx.types.uint(256);
    AddressCode            => sym::code
                           => gcx.types.bytes_ref.memory;
    AddressCodehash        => sym::codehash
                           => gcx.types.fixed_bytes(32);
    AddressCall            => kw::Call
                           => gcx.mk_builtin_fn(&[gcx.types.bytes_ref.memory], SM::View, &[gcx.types.bytes_ref.memory]);
    AddressDelegatecall    => kw::Delegatecall
                           => gcx.mk_builtin_fn(&[gcx.types.bytes_ref.memory], SM::View, &[gcx.types.bytes_ref.memory]);
    AddressStaticcall      => kw::Staticcall
                           => gcx.mk_builtin_fn(&[gcx.types.bytes_ref.memory], SM::View, &[gcx.types.bytes_ref.memory]);

    AddressPayableTransfer => sym::transfer
                           => gcx.mk_builtin_fn(&[gcx.types.uint(256)], SM::NonPayable, &[]);
    AddressPayableSend     => sym::send
                           => gcx.mk_builtin_fn(&[gcx.types.uint(256)], SM::NonPayable, &[gcx.types.bool]);

    FixedBytesLength       => sym::length
                           => gcx.types.uint(8);

    ArrayLength            => sym::length
                           => gcx.types.uint(256);

    ErrorSelector          => sym::selector
                           => gcx.types.fixed_bytes(4);

    EventSelector          => sym::selector
                           => gcx.types.fixed_bytes(32);

    // `type(T)`
    ContractCreationCode   => sym::creationCode
                           => gcx.types.bytes_ref.memory;
    ContractRuntimeCode    => sym::runtimeCode
                           => gcx.types.bytes_ref.memory;
    ContractName           => sym::name
                           => gcx.types.string_ref.memory;
    InterfaceId            => sym::interfaceId
                           => gcx.types.fixed_bytes(4);
    TypeMin                => sym::min => unreachable!();
    TypeMax                => sym::max => unreachable!();

    // `TyKind::Type` (`string.concat`, on the `string` type, not a string value)
    UdvtWrap               => sym::wrap   => unreachable!();
    UdvtUnwrap             => sym::unwrap => unreachable!();

    // TODO                => `(string memory...) pure returns(string memory)`
    StringConcat           => sym::concat
                           => gcx.mk_builtin_fn(&[], SM::Pure, &[gcx.types.string_ref.memory]);

    // TODO                => `(bytes memory...) pure returns(bytes memory)`
    BytesConcat            => sym::concat
                           => gcx.mk_builtin_fn(&[], SM::Pure, &[gcx.types.bytes_ref.memory]);
}

impl Builtin {
    const FIRST_GLOBAL: usize = 0;
    const LAST_GLOBAL: usize = Self::Abi as usize + 1;

    const FIRST_BLOCK: usize = Self::BlockCoinbase as usize;
    const LAST_BLOCK: usize = Self::BlockBlobbasefee as usize + 1;

    const FIRST_MSG: usize = Self::MsgSender as usize;
    const LAST_MSG: usize = Self::MsgSig as usize + 1;

    const FIRST_TX: usize = Self::TxOrigin as usize;
    const LAST_TX: usize = Self::TxGasPrice as usize + 1;

    const FIRST_ABI: usize = Self::AbiEncode as usize;
    const LAST_ABI: usize = Self::AbiDecode as usize + 1;

    /// Returns an iterator over all builtins.
    #[inline]
    pub fn iter() -> std::iter::Map<std::ops::Range<usize>, impl FnMut(usize) -> Self> {
        (0..Self::COUNT).map(|i| Self::from_index(i).unwrap())
    }

    #[inline]
    const fn from_index(i: usize) -> Option<Self> {
        const {
            assert!(Self::COUNT <= Primitive::MAX as usize);
            assert!(size_of::<Self>() == 1);
        }
        if i < Self::COUNT {
            // SAFETY:
            //
            // `Self` is a field-less, `repr(Primitive)` enum and therefore guaranteed
            // to have the same size and alignment as `Primitive`.
            //
            // This branch ensures `i < Self::COUNT` where `Self::COUNT` is the
            // number of variants in `Self`. The discriminants of `Self` are
            // contiguous because no variant specifies a custom discriminant
            // with `Variant = value`. This ensures that `i as Primitive` is
            // a valid inhabitant of type `Self`.
            Some(unsafe { std::mem::transmute::<Primitive, Self>(i as Primitive) })
        } else {
            None
        }
    }

    /// Returns the global builtins.
    pub fn global() -> impl ExactSizeIterator<Item = Self> + Clone {
        Self::make_range_iter(Self::FIRST_GLOBAL..Self::LAST_GLOBAL)
    }

    /// Returns the builtin's members.
    pub fn members(self) -> Option<impl ExactSizeIterator<Item = Self> + Clone> {
        use Builtin::*;
        Some(Self::make_range_iter(match self {
            Block => Self::FIRST_BLOCK..Self::LAST_BLOCK,
            Msg => Self::FIRST_MSG..Self::LAST_MSG,
            Tx => Self::FIRST_TX..Self::LAST_TX,
            Abi => Self::FIRST_ABI..Self::LAST_ABI,
            _ => return None,
        }))
    }

    #[inline]
    fn make_range_iter(
        range: std::ops::Range<usize>,
    ) -> impl ExactSizeIterator<Item = Self> + Clone {
        debug_assert!(range.start < Self::COUNT);
        debug_assert!(range.end < Self::COUNT);
        (range.start as Primitive..range.end as Primitive)
            .map(|idx| unsafe { Self::from_index(idx as usize).unwrap_unchecked() })
    }
}
